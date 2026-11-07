use yesod::direction::project_line;
use yesod::rand::{pcg3df, r2};

use super::*;
use crate::algorithm::rc::{Ray, RayComps, RcData};
use crate::cache::LoadedNode;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
pub struct Hit {
    pub position: Vec3<f32>,
    pub normal: Vec3<f32>,
    pub radiance: Vec3<f32>,
}
impl HitExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.position.x.is_finite()
    }
}
impl Hit {
    pub fn invalid() -> Self {
        Self {
            position: Vec3::splat(f32::INFINITY),
            normal: Vec3::splat(0.0),
            radiance: Vec3::splat(0.0),
        }
    }
}

pub struct TraceData {
    // Number of rays traced through this probe
    pub ray_count: Buffer<u32>,
    // Next offset for the highest cascade's ray directions
    pub next_max_cascade_ray_direction: Singleton<u32>,
    // Offsets for ray directions for each probe
    pub ray_offset: Buffer<u32>,
    // Index of current ray direction within a base probe
    pub base_ray_index: Buffer<u32>,
    pub traced_hits: Buffer<Hit>,
    pub traced_rays: Buffer<Ray>,
    pub next_traced_ray: Singleton<u32>,
}

impl TraceData {
    pub fn new(core: &CoreData, rc: &RcData, max_rays: usize) -> Self {
        Self {
            ray_count: DEVICE.create_buffer_from_fn::<u32>(rc.cache.total_probes() as usize, |_| 0),
            next_max_cascade_ray_direction: Singleton::new(0),
            ray_offset: DEVICE
                .create_buffer_from_fn::<u32>(rc.cache.total_probes() as usize, |_| 0),
            base_ray_index: DEVICE
                .create_buffer_from_fn::<u32>(rc.cache.base_probes as usize, |_| 0),
            traced_hits: DEVICE
                .create_buffer_from_fn::<Hit>(core.num_pixels() as usize, |_| Hit::invalid()),
            traced_rays: DEVICE.create_buffer::<Ray>(max_rays),
            next_traced_ray: Singleton::new(0),
            // used for debug
        }
    }
    pub fn compute_ray_count_base(&self, core: &CoreData, rc: &RcData) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([8, 8, 1]);
                let world_pos = core.get_world_pos(dispatch_id().xy());

                let lod = rc.cache.estimate_lod(world_pos);

                let probe = rc.cache.quantize(world_pos, lod, 0.expr());

                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }

                self.ray_count
                    .atomic_ref(node.index)
                    .fetch_add(1_u32.expr());
            }),
            core.screen_dispatch_size(),
        )
        .debug("Compute Ray Count Cascade 0")
    }
    pub fn compute_ray_count(&self, rc: &RcData, cascade: u32) -> NodeConfigs<'static> {
        dispatch1(
            track!(|cascade| {
                let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                if node.invalid() {
                    return;
                }
                let next = node.probe.next_nearest();
                let next = rc.cache.get(next); // must exist, unless the table overflowed
                if next.invalid() {
                    return;
                }

                self.ray_count
                    .atomic_ref(next.index)
                    .fetch_add(self.ray_count.read(node.index));
            }),
            [rc.cache.max_probes_at_cascade(cascade - 1), 1, 1],
            cascade - 1,
        )
        .debug(format!("Compute Ray Count Cascade {cascade}"))
    }
    pub fn compute_ray_offset(&self, rc: &RcData, cascade: u32) -> NodeConfigs<'static> {
        if cascade == rc.last_cascade() {
            dispatch0(
                track!(|| {
                    let node = rc.cache.get_occupied(cascade.expr(), dispatch_id().x);
                    if node.invalid() {
                        return;
                    }
                    let num_rays = self.ray_count.read(node.index);
                    self.ray_offset.write(
                        node.index,
                        self.next_max_cascade_ray_direction
                            .atomic()
                            .fetch_add(num_rays),
                    );
                }),
                [rc.cache.max_probes_at_cascade(cascade), 1, 1],
            )
        } else {
            dispatch1(
                track!(|cascade| {
                    let node = rc.cache.get_occupied(cascade, dispatch_id().x);
                    if node.invalid() {
                        return;
                    }
                    let lower_nodes = <[LoadedNode; 8]>::var_zeroed();
                    let counts = <[u32; 8]>::var_zeroed();
                    let j = 0_u32.var();
                    for i in 0_u32..8_u32 {
                        let lower_probe = node.probe.last_nearest(i);
                        let lower_node = rc.cache.get(lower_probe);
                        if lower_node.loaded() {
                            lower_nodes.write(j, lower_node);
                            counts.write(j, self.ray_count.read(lower_node.index));
                            *j += 1;
                        }
                    }
                    let offset = self.ray_offset.read(node.index).var();
                    for i in 0.expr()..**j {
                        self.ray_offset.write(lower_nodes.read(i).index, offset);
                        *offset += counts.read(i);
                    }
                }),
                [rc.cache.max_probes_at_cascade(cascade + 1), 1, 1],
                cascade + 1,
            )
        }
        .debug(format!("Compute Ray Offset Cascade {cascade}"))
    }
    pub fn compute_rays(
        &self,
        core: &CoreData,
        rc: &RcData,
        tick: u32,
        c0_len: f32,
        multibounce: bool,
    ) -> NodeConfigs<'static> {
        memo(multibounce, || {
            DEVICE.create_kernel::<fn(u32, f32)>(&track!(|tick, c0_len| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();
                let world_pos = core.get_world_pos(pixel);
                let normal = core.read_normal(pixel);

                let lod = rc.cache.estimate_lod(world_pos);
                let probe = rc.cache.quantize(world_pos, lod, 0.expr());
                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }

                let offset = self.ray_offset.read(node.index);

                let offset_index = offset + self.base_ray_index.atomic_ref(node.index).fetch_add(1);

                let dir = project_line(
                    (r2(offset_index) + pcg3df(Vec3::expr(tick, 29, 968)).xy()).fract(),
                );
                let dir = dir * dir.dot(normal).signum();

                let start = offset_ray(world_pos, normal);
                let hit = core.world.trace_segment(
                    start,
                    dir,
                    Vec2::expr(rc.cm1_len * rc.cache.spacing(lod, 0), f32::INFINITY),
                );

                let radiance = core.get_radiance(multibounce.then_some(rc), start, dir, hit);

                self.traced_hits.write(
                    core.pixel_to_linear(pixel),
                    if hit.miss() {
                        Hit::invalid().expr()
                    } else {
                        Hit::from_comps_expr(HitComps {
                            position: start + dir * hit.committed_ray_t,
                            normal: core.world.get_normal_flat(hit, dir),
                            radiance,
                        })
                    },
                );

                rc.cache.iter_lods(world_pos, |lod, _, _| {
                    self.compute_deposit_ray(rc, world_pos, lod, dir, radiance, hit, c0_len);
                })
            }))
        })
        .dispatch_async(core.screen_dispatch_size(), &tick, &c0_len)
        .debug("Compute Rays")
    }
    pub fn compute_secondary_rays(
        &self,
        core: &CoreData,
        rc: &RcData,
        tick: u32,
        c0_len: f32,
        multibounce: bool,
    ) -> NodeConfigs<'static> {
        memo(multibounce, || {
            DEVICE.create_kernel::<fn(u32, f32)>(&track!(|tick, c0_len| {
                let hit = self.traced_hits.read(dispatch_id().x);
                if !hit.valid() {
                    return;
                }
                let world_pos = hit.position;
                let normal = hit.normal;

                let lod = rc.cache.estimate_lod_secondary(world_pos);
                let probe = rc.cache.quantize(world_pos, lod, 0.expr());
                let node = rc.cache.get(probe);

                if node.invalid() {
                    device_println!("Should be impossible!");
                    return;
                }
                let dir = project_line(pcg3df(Vec3::expr(dispatch_id().x, tick, 173)).xy());
                let dir = dir * dir.dot(normal).signum();
                let start = offset_ray(world_pos, normal);

                let hit = core.world.trace_segment(
                    start,
                    dir,
                    Vec2::expr(rc.cm1_len * rc.cache.spacing(lod, 0), f32::INFINITY),
                );

                let radiance = core.get_radiance(multibounce.then_some(rc), start, dir, hit);

                self.compute_deposit_ray(rc, world_pos, lod, dir, radiance, hit, c0_len);
            }))
        })
        .dispatch_async([self.traced_hits.len() as u32, 1, 1], &tick, &c0_len)
        .debug("Secondary: Compute Rays")
    }
    #[tracked]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_deposit_ray(
        &self,
        rc: &RcData,
        world_pos: Expr<Vec3<f32>>,
        lod: Expr<u32>,
        dir: Expr<Vec3<f32>>,
        radiance: Expr<Vec3<f32>>,
        hit: Expr<crate::algorithm::Hit>,
        c0_len: Expr<f32>,
    ) {
        let probe = rc.cache.quantize(world_pos, lod, 0.expr());

        let ray_index = self.next_traced_ray.atomic().fetch_add(1_u32.expr());

        if ray_index >= self.traced_rays.len_expr_u32() {
            return;
        }

        let cascade = if hit.miss() {
            rc.last_cascade().expr()
        } else {
            let cascade = keter::max(
                (hit.committed_ray_t / (c0_len * rc.cache.spacing(lod, 0))).log(rc.length_scale),
                0.0,
            )
            .ceil()
            .cast_u32();

            keter::min(cascade, rc.last_cascade())
        };

        let ray = Ray::from_comps_expr(RayComps {
            radiance,
            base_probe: probe,
            cascade,
            dir: SphericalEncoder::encode(dir),
        });
        self.traced_rays.write(ray_index, ray);
    }
    pub fn reset(&self) -> NodeConfigs<'static> {
        (
            dispatch0(
                track!(|| {
                    self.ray_count.write(dispatch_id().x, 0_u32);
                }),
                [self.ray_count.len() as u32, 1, 1],
            )
            .debug("Reset Node Ray Count"),
            dispatch0(
                track!(|| {
                    self.base_ray_index.write(dispatch_id().x, 0_u32);
                }),
                [self.base_ray_index.len() as u32, 1, 1],
            )
            .debug("Reset Base Ray Offsets"),
            dispatch0(
                track!(|| {
                    self.next_traced_ray.write(0_u32);
                    self.next_max_cascade_ray_direction.write(0_u32);
                }),
                [1, 1, 1],
            )
            .debug("Reset Next Traced Ray"),
        )
            .chain()
    }
}
