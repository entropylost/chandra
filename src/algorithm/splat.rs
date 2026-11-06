use keter::lang::functions::{block_id, sync_block, thread_id, warp_read_lane_at};
use yesod::color::inferno;
use yesod::direction::OctahedralEncoder;
use yesod::shapes::intersect_sphere;
use yesod::utils::FetchAddVector;

use super::*;
use crate::algorithm::rc::RcData;
use crate::cache::{DirectionIndexer, DirectionIndexerComps, ProbeComps};

pub struct SplatData {
    pub radiance: Buffer<Vec4<f32>>,
    // debug only
    pub depth: Buffer<f32>,
    pub cm1_shadow_mask: Buffer<u32>,
    // for optimized 32-dir
    pub irradiance: Tex2dHandle<Vec4<f32>>,
    // settings
    pub interpolation_exponent: f32,
    pub irradiance_resolution: u32,
    pub irradiance_dims: Vec2<u32>,
}

impl SplatData {
    pub fn new(core: &CoreData, rc: &RcData, interpolation_exponent: f32) -> Self {
        let c0_size = rc.cache.base_probes;
        let x_size = 1 << (c0_size.ilog2() / 2);
        let y_size = c0_size.div_ceil(x_size);
        let resolution = 6;
        let full_resolution = resolution + 2;
        let irradiance_texture = DEVICE.create_tex2d::<Vec4<f32>>(
            PixelStorage::Half4,
            x_size * full_resolution,
            y_size * full_resolution,
            1,
        );
        let irradiance = BINDLESS.push_tex2d(
            irradiance_texture,
            Sampler {
                filter: SamplerFilter::LinearLinear,
                address: SamplerAddress::Edge,
            },
        );
        Self {
            radiance: DEVICE.create_buffer::<Vec4<f32>>(core.num_pixels() as usize),
            depth: DEVICE.create_buffer::<f32>(core.num_pixels() as usize),
            cm1_shadow_mask: DEVICE.create_buffer::<u32>(core.num_pixels() as usize),
            irradiance,
            interpolation_exponent,
            irradiance_resolution: resolution,
            irradiance_dims: Vec2::new(x_size, y_size),
        }
    }
    fn irradiance_size(&self) -> u32 {
        self.irradiance_resolution + 2
    }
    pub fn calculate_irradiance(&self, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([1, 32, 1]);
                assert!(rc.increased_base_resolution);
                assert_eq!(rc.base_angular_resolution, 4);

                let node = rc.cache.get_occupied(0.expr(), dispatch_id().x);
                if node.invalid() {
                    return;
                }

                // Each thread loads a direction of the probe, then gathers texels of the irradiance map from all
                // the directions, which takes fewer instructions than summing every texel across the warp.
                let radiances = Shared::<Vec4<f32>>::new(32);
                let dirs = Shared::<Vec4<f32>>::new(32);

                let radiance = rc.base_radiance.read(node.data_offset + dispatch_id().y);
                let radiance = if radiance.is_finite().all() {
                    radiance.extend(1.0)
                } else {
                    Vec4::splat_expr(0.0)
                };
                let dir = rc
                    .cache
                    .get_encoder(0.expr())
                    .linear_to_dir(dispatch_id().y);
                radiances.write(dispatch_id().y, radiance);
                dirs.write(dispatch_id().y, dir.extend(0.0));
                sync_block();

                let tile = Vec2::expr(
                    dispatch_id().x % self.irradiance_dims.x,
                    dispatch_id().x / self.irradiance_dims.x,
                );
                // Every texel of the tile, its border included, so the borders need no copy pass. With
                // 6x6 texels the tile has 64, which takes as many passes over the warp as the 36 inside.
                let res = self.irradiance_resolution;
                let size = self.irradiance_size();
                let texels = size * size;
                for pass in 0..texels.div_ceil(32) {
                    let index = dispatch_id().y + pass * 32;
                    if index < texels {
                        let texel = Vec2::expr(index % size, index / size);
                        // The octahedral map mirrors across each edge: a border texel left or right of the
                        // interior continues it at the mirrored row, one above or below at the mirrored
                        // column, and a corner at the opposite corner of the interior.
                        let source = if texel.x == 0 || texel.x == res + 1 {
                            let x = if texel.x == 0 { 1_u32.expr() } else { res.expr() };
                            Vec2::expr(x, res + 1 - texel.y)
                        } else {
                            texel
                        };
                        let source = if source.y == 0 || source.y == res + 1 {
                            let y = if source.y == 0 { 1_u32.expr() } else { res.expr() };
                            Vec2::expr(res + 1 - source.x, y)
                        } else {
                            source
                        };
                        let offset = (source.cast_f32() - 0.5) / res as f32;
                        let out_dir = OctahedralEncoder::decode(offset);
                        let total = Vec4::splat(0.0_f32).var();
                        for i in 0..32_u32 {
                            let weight = keter::max(out_dir.dot(dirs.read(i).xyz()), 0.0);
                            *total += radiances.read(i) * weight;
                        }
                        let uv = tile * size + texel;
                        self.irradiance.write(uv, **total);
                    }
                }
            }),
            [rc.max_probes[0], 32, 1],
        )
        .debug("Calculate Irradiance")
    }
    #[tracked]
    fn get_lambertian_weight(
        &self,
        encoder: Expr<DirectionIndexer>,
        dir: Expr<u32>,
        normal: Expr<Vec3<f32>>,
    ) -> Expr<f32> {
        let dir = encoder.linear_to_grid(dir).cast_f32();
        let total_weight = 0.0f32.var();
        for i in 0u32..16 {
            let offset = Vec2::expr(i % 4, i / 4);
            let subdir = encoder.grid_to_dir_f32(dir + (offset.cast_f32() + 0.5) / 4.0);
            *total_weight += keter::max(subdir.dot(normal), 0.0);
        }
        let output = total_weight / 16.0;
        output
    }

    // TODO: Can weigh which LOD to interpolate from based on quality.
    pub fn shade_optimized_32(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([4, 1, 8]);
                let pixel = dispatch_id().xy();
                let thread = thread_id().z;
                let world_pos = core.get_world_pos(pixel);

                let result = Vec4::splat(0.0f32).var();

                rc.cache.iter_lods(world_pos, |lod, _, lod_weight| {
                    let interp_probes = rc.cache.quantize_trilinear(world_pos, lod, 0.expr());

                    let normal = core.read_normal(pixel);
                    let dir = OctahedralEncoder::encode(normal);
                    let dir = dir * self.irradiance_resolution as f32 + 1.0;

                    let (probe, weight) = interp_probes.get(thread);
                    let probe = Probe::from_comps_expr(ProbeComps {
                        position: probe.extend(lod),
                        cascade: 0.expr(),
                    });
                    let node = rc.cache.get(probe);
                    if node.loaded() {
                        let tile = Vec2::expr(
                            node.index % self.irradiance_dims.x,
                            node.index / self.irradiance_dims.x,
                        );
                        let uv = (tile * self.irradiance_size()).cast_f32() + dir;
                        let irradiance = self.irradiance.var().sample(
                            uv / (self.irradiance_dims * self.irradiance_size())
                                .expr()
                                .cast_f32(),
                        );
                        *result += irradiance * weight * lod_weight * rc.get_quality_weight(node);
                    }
                });

                for i in 0u32..3 {
                    if thread % (1 << i) == 0 {
                        *result +=
                            warp_read_lane_at(**result, thread_id().x + 4 * (thread + (1 << i)));
                    }
                }

                if thread == 0 {
                    let pixel_index = core.pixel_to_linear(pixel);
                    self.radiance.write(pixel_index, result);
                }
            }),
            [core.screen_size().x, core.screen_size().y, 8],
        )
        .debug("Draw Splats (Optimized, 32 dirs)")
    }
    pub fn shade_optimized_8(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([4, 1, 8]);
                assert!(rc.increased_base_resolution);
                assert_eq!(rc.base_angular_resolution, 2);

                let pixel = dispatch_id().xy();
                // assume there's no early-exit for now.
                let world_pos = core.get_world_pos(pixel);

                let result = Vec4::splat(0.0_f32).var();

                let thread = thread_id().z;

                let normal = core.read_normal(pixel);
                let encoder = rc.cache.get_encoder(0.expr());

                let direction_weight = self.get_lambertian_weight(encoder, thread, normal);

                let cm1_shadow_mask = if rc.cm1_len > 0.0 {
                    self.cm1_shadow_mask.read(core.pixel_to_linear(pixel))
                } else {
                    0u32.expr()
                };

                rc.cache.iter_lods(world_pos, |lod, _, lod_weight| {
                    let interp_probes = rc.cache.quantize_trilinear(world_pos, lod, 0.expr());

                    /*
                    let thread = thread_id().z;

                    let adj_offsets = Shared::<u32>::new(32);
                    let adj_weights = Shared::<f32>::new(32);
                    let pixel_offsets = Shared::<u32>::new(5);

                    let (probe, weight) = interp_probes.get(i);
                    let probe = Probe::from_comps_expr(ProbeComps {
                        position: probe.extend(lod),
                        cascade: 0.expr(),
                    });
                    let node = rc.cache.get(probe);
                    if node.loaded() {
                        let index = warp_prefix_count_bits(true.expr());
                        adj_offsets.write(index, node.data_offset);
                        adj_weights.write(index, weight);
                        if thread == 0 {
                            pixel_offsets.write(thread_id().x, index);
                        }
                    }
                    let n = warp_active_count_bits(next.loaded());
                    pixel_offsets.write(4, n);

                    let normal = core.read_normal(pixel);
                    let encoder = rc.cache.get_encoder(0.expr());

                    let dir = encoder.linear_to_dir(thread);
                    let weight = dir.dot(normal);

                    let valid_dirs = Shared::<u32>::new(32);
                    if weight > 0.0 {
                        valid_dirs.write(index, dir);
                    }
                    */

                    for i in 0u32..8 {
                        let (probe, weight) = interp_probes.get(i);
                        let probe = Probe::from_comps_expr(ProbeComps {
                            position: probe.extend(lod),
                            cascade: 0.expr(),
                        });
                        let node = rc.cache.get(probe);
                        if node.invalid() {
                            continue;
                        }
                        let radiance = rc.base_radiance.read(node.data_offset + thread);
                        let radiance = if cm1_shadow_mask & (1 << thread) == 0 {
                            radiance
                        } else {
                            Vec3::splat_expr(0f32)
                        };
                        if radiance.is_finite().all() {
                            *result += radiance.extend(1.0)
                                * weight
                                * lod_weight
                                * rc.get_quality_weight(node);
                        }
                    }
                });
                *result *= direction_weight;
                for i in 0u32..3 {
                    if thread % (1 << i) == 0 {
                        *result +=
                            warp_read_lane_at(**result, thread_id().x + 4 * (thread + (1 << i)));
                    }
                }

                if thread == 0 {
                    let pixel_index = core.pixel_to_linear(pixel);
                    self.radiance.write(pixel_index, result);
                }
            }),
            [core.screen_size().x, core.screen_size().y, 8],
        )
        .debug("Draw Splats (Optimized, 8 dirs)")
    }
    #[tracked]
    pub fn get_radiance(&self, pixel_index: Expr<u32>) -> Expr<Vec3<f32>> {
        let radiance = self.radiance.read(pixel_index);
        let radiance = radiance.xyz() / keter::max(radiance.w, 0.00001);
        radiance.powf(self.interpolation_exponent.recip())
    }
    pub fn trace_cm1(
        &self,
        core: &CoreData,
        rc: &RcData,
        num_samples: u32,
    ) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();

                let view = core.view.read();
                let world_pos = core.get_world_pos(pixel);
                let lod = rc.cache.estimate_lod(world_pos);

                let cm1_len = rc.cm1_len * rc.cache.base_grid_size * (1_u32 << lod).cast_f32();

                let shadow_mask = 0u32.var();
                // TODO: Adjust for less base resolution.
                let encoder = self.get_encoder(rc, 0.expr());

                for i in 0_u32.expr()..encoder.linear_size() {
                    let offset = pcg3df(pixel.extend(i)).xy();
                    let cm1_dir =
                        encoder.grid_to_dir_f32(encoder.linear_to_grid(i).cast_f32() + offset);
                    let cm1_dir = cm1_dir * cm1_len;
                    // raymarch in screenspace
                    for j in 0_u32..num_samples {
                        let sample_pos =
                            world_pos + cm1_dir * ((j.cast_f32() + 0.5) / num_samples as f32);
                        // does this add the 0.5 bias anyways?
                        let screen_pos = view.project(sample_pos);
                        let sample_depth = (sample_pos - view.pos).length();
                        let object_depth = core.depth.read(screen_pos.cast_u32());
                        if object_depth != 0.0
                            && object_depth < sample_depth * 0.9992
                            && !(sample_depth > object_depth + cm1_len)
                        {
                            *shadow_mask |= 1 << i;
                            break;
                        }
                    }
                }
                self.cm1_shadow_mask
                    .write(core.pixel_to_linear(pixel), shadow_mask);
            }),
            core.screen_dispatch_size(),
        )
        .debug("Trace Cascade -1")
    }
    #[tracked]
    fn get_encoder(&self, rc: &RcData, cascade: Expr<u32>) -> Expr<DirectionIndexer> {
        DirectionIndexer::from_comps_expr(DirectionIndexerComps {
            resolution: if rc.increased_base_resolution && cascade == 0 {
                rc.cache.base_angular_resolution.expr()
            } else {
                (rc.cache.base_angular_resolution / 2) * (1 << cascade)
            },
        })
    }
    pub fn splat_radiance_raw(
        &self,
        core: &CoreData,
        rc: &RcData,
    ) -> &'static Kernel<fn(f32, u32, u32)> {
        memo(
            rc.increased_base_resolution,
            track!(|| {
                DEVICE.create_kernel::<fn(f32, u32, u32)>(&|blend_radius, cascade, draw_mode| {
                    set_block_size([1, 32, 1]);
                    let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                    if node.invalid() {
                        return;
                    }
                    let view = core.view.read();

                    let probe = node.probe;
                    let encoder = self.get_encoder(rc, cascade);

                    // let quality = node_quality.read(node.index);

                    // 8 if !INCREASED_BASE_RESOLUTION

                    let radiance_data = Shared::<Vec3<f32>>::new(32);
                    if rc.increased_base_resolution
                        && cascade == 0
                        && dispatch_id().y < encoder.linear_size()
                    {
                        let v = rc.base_radiance.read(node.data_offset + dispatch_id().y);
                        radiance_data.write(thread_id().y, v);
                    } else if cascade != 0 || dispatch_id().y < encoder.linear_size() {
                        let v = rc.radiance.read(node.data_offset / 4 + dispatch_id().y);
                        radiance_data.write(thread_id().y, v);
                    }
                    sync_block();

                    let screen_min = core.screen_size().expr().cast_f32().var();
                    let screen_max = Vec2::splat(0.0_f32).var();

                    let spacing = rc.cache.probe_spacing(probe) * blend_radius;

                    /*if dispatch_id().y < 8 {
                        let i = dispatch_id().y;

                        let coord = probe_pos
                            + spacing
                                * (Vec3::<bool>::expr((i & 1) != 0, (i & 2) != 0, (i & 4) != 0)
                                    .select(Vec3::splat_expr(-1.0_f32), Vec3::splat_expr(1.0_f32)));
                        let screen_pos = view.project3(coord);
                        if screen_pos.z <= 0.0 {
                            return;
                        }
                        let screen_pos = screen_pos.xy();

                        screen_min = warp_active_min(screen_pos);
                        screen_max = warp_active_max(screen_pos);
                    }*/

                    let pos = if draw_mode == 0 {
                        rc.cache.dequantize(probe)
                    } else {
                        rc.average_position.read(node.index).position()
                    };

                    // TODO: May go schizo if the probe is behind the camera...
                    for i in 0..8 {
                        let coord = pos
                            + spacing
                                * (Vec3::<bool>::expr((i & 1) != 0, (i & 2) != 0, (i & 4) != 0)
                                    .select(Vec3::splat_expr(-1.0_f32), Vec3::splat_expr(1.0_f32)));
                        let screen_pos = view.project3(coord);
                        if screen_pos.z <= 0.0 {
                            return;
                        }
                        let screen_pos = screen_pos.xy();
                        *screen_min = keter::min(**screen_min, screen_pos);
                        *screen_max = keter::max(**screen_max, screen_pos);
                    }
                    let screen_min = keter::max(0.0_f32, **screen_min).cast_u32();
                    let screen_max = keter::min(core.screen_size().expr().cast_f32(), **screen_max)
                        .ceil()
                        .cast_u32();
                    let size = screen_max - screen_min;
                    let area = size.reduce_prod();
                    let i = dispatch_id().y.var();
                    while i < area {
                        let pixel = screen_min + Vec2::expr(i % size.x, i / size.x);
                        let pixel_index = core.pixel_to_linear(pixel);
                        *i += 32;

                        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

                        // TODO: This is a complete fucking mess and needs a lot of cleanup.
                        if draw_mode == 0 {
                            let depth = core.read_depth(pixel);
                            if depth == 0.0 {
                                continue;
                            }
                            let world_pos = view.pos + ray_dir * depth;
                            let normal = core.read_normal(pixel);
                            let trilinear_weight =
                                keter::max(1.0 - (world_pos - pos).abs() / spacing, 0.0)
                                    .reduce_prod();
                            let weight = trilinear_weight * rc.get_quality_weight(node);
                            // * (1 << (2 * probe.position.w)).cast_f32().recip()
                            if weight == 0.0 {
                                continue;
                            }

                            let cm1_shadow_mask = if rc.cm1_len > 0.0 && cascade == 0 {
                                self.cm1_shadow_mask.read(pixel_index)
                            } else {
                                0u32.expr()
                            };

                            let total_radiance = Vec4::splat(0.0_f32).var();
                            for j in 0_u32.expr()..encoder.linear_size() {
                                let weight = self.get_lambertian_weight(
                                    encoder,
                                    j + block_id().y * 32,
                                    normal,
                                );
                                if weight > 0.0 {
                                    let radiance = if cm1_shadow_mask & (1 << j) == 0 {
                                        radiance_data.read(j)
                                    } else {
                                        Vec3::splat_expr(0f32)
                                    };
                                    if radiance.is_finite().all() {
                                        *total_radiance += weight * radiance.extend(1.0);
                                        // } else {
                                        //     *total_radiance.w = 0.0;
                                        //     break;
                                    }
                                }
                            } // TODO: Figure out the biasing properly? Just directly bias based on amount of missing data?
                            if total_radiance.w < 0.0001 {
                                continue;
                            }

                            let total_radiance = total_radiance.xyz() / total_radiance.w;
                            // let total_radiance = pcg3df(probe.position.xyz());
                            let deposit_radiance =
                                total_radiance.powf(self.interpolation_exponent).extend(1.0)
                                    * weight;
                            let dest = self.radiance.atomic_ref(pixel_index);
                            dest.fetch_add(deposit_radiance);
                        } else if draw_mode == 1 || draw_mode == 2 {
                            // TODO: What about negative interesctions?
                            let (time, intersected) =
                                intersect_sphere(view.pos - pos, ray_dir, spacing);
                            if !intersected {
                                continue;
                            }
                            let depth = time.x;
                            let intersect_pos = view.pos + ray_dir * depth;

                            let intersect_dir = (intersect_pos - pos).normalize();
                            let index = encoder.dir_to_linear(intersect_dir) - block_id().y * 32;
                            if index >= 32 {
                                continue;
                            }

                            if draw_mode == 2 {
                                // depth prepass
                                self.depth.atomic_ref(pixel_index).fetch_min(depth);
                            } else if draw_mode == 1 {
                                let actual_depth = self.depth.read(pixel_index);
                                if depth == actual_depth {
                                    // let radiance = radiance_data.read(index);
                                    let weight = rc.cache.weights.read(
                                        node.data_offset
                                            + rc.cache
                                                .get_encoder(cascade)
                                                .dir_to_linear(intersect_dir),
                                    );
                                    self.radiance
                                        .write(pixel_index, inferno(weight / 10.0).extend(1.0));
                                    // if radiance.is_finite().all() {
                                    //     let deposit_radiance =
                                    //         radiance.powf(self.interpolation_exponent).extend(1.0);
                                    //     self.radiance.write(pixel_index, deposit_radiance);
                                    // }
                                }
                            }
                        }
                    }
                })
            }),
        )
    }

    pub fn reset(&self) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                self.radiance.write(dispatch_id().x, Vec4::splat(0.0_f32));
                self.depth.write(dispatch_id().x, f32::INFINITY);
            }),
            [self.radiance.len() as u32, 1, 1],
        )
        .debug("Reset Radiance Buffer")
    }
}
