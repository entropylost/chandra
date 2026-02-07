#![feature(array_windows)]

/*
TODOS:
[X] Add more lights, fancier primary ray stuff?
[ ] Optimize, switch to u64 everywhere
[ ] Better hashmap impl without dropping things? Bias towards primary rays first.
[ ] Some sorta anderson acceleration to make the infinite bounce convergence less obvious?

[ ] Precompute ndotl lighting separately? At least for offscreen?
[ ] Add separate world-space cache with lower resolution? Preferably get both to use the same map to prevent overflow...
[ ] Ray counting / quality metrics.
[ ] Test how much memory saved if each probe element is sparse?

[X] Test completely normal RC implementation for performance
[ ] Compress probe lists. Then can use bigger hashmap.
  - Could use a freelist to avoid duplication, or just double everything.
[ ] Need to fix LODs. How??
  [ ] Write every ray to two LODs and have perfect blending.
[ ] Merging optimization..
[ ] Convert to SH at C0?
[ ] Alternative final-level gathering?

[ ] Fix the ray start bias to actually use stuff from that paper.
[ ] Fix the shadow fix causing overdarkening / inconsistency between cascades.
[ ] Init cascades upon screenspace rather than rays and merge upwards.
[ ] Investigate light leaking near pillars without shadow fix

BUGS:
[ ] Convergence is just pretty damn slow currently. Make it so that lower cascades use higher cascade values if they don't have enough info?

Split into two caches. Also measure ray divergence and use that to stochastically reduce ray count.
*/

use std::f32::consts::TAU;
use std::path::PathBuf;

use chandra::cache::{
    Fluence, HashCache, INDEX_SCALING, LoadedNode, LoadedNodeComps, Node, POSITION_INVALID, Probe,
    ProbeComps, Radiance, RadianceDeposit, RadianceElement, decode_position,
};
use chandra::world::TriangleWorld;
use glam::Mat3 as FMat3;
use image::{GenericImageView, ImageReader};
use keter::bindless::BINDLESS;
use keter::graph::profile::Profiler;
use keter::lang::functions::sync_block;
use keter::lang::types::shared::Shared;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter::rtx::SurfaceHit;
use keter_testbed::{App, KeyCode};
use yesod::camera::{Camera, View};
use yesod::color::{inferno, plasma, reinhard};
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::printer::global::flush_printer;
use yesod::rand::{pcg3d, pcg3df};
use yesod::{device_println, host};

const TERMINATE_DEPOSIT: u32 = 0b1;
// TODO: Probably should scale with distance

const TARGET_QUALITY_MULTIPLIER: f32 = 50.0;
const QUALITY_DECAY: f32 = 0.95;
const MAX_RAYS: u32 = 1 << 22;
const NODE_LIFETIME: u32 = 4;

fn sunlight_dir() -> Vec3<f32> {
    let dir = glam::Vec3::new(0.1, 0.5, 1.0);
    dir.normalize().into()
}

#[tracked]
fn reflect(dir: Expr<Vec3<f32>>, normal: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    dir - normal * (2.0_f32.expr() * dir.dot(normal))
}

#[tracked]
fn rotate_hemi(normal: Expr<Vec3<f32>>, v: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    // Find perpendicular basis; due to hairy ball theorem this cannot be continuous.
    let perp1 = if normal.z.abs() < 0.9 {
        normal.cross(Vec3::expr(0.0_f32, 0.0, 1.0)).normalize()
    } else {
        normal.cross(Vec3::expr(0.0_f32, 1.0, 0.0)).normalize()
    };
    let perp2 = normal.cross(perp1);
    v.x * perp1 + v.y * perp2 + v.z * normal
}

// https://ameye.dev/notes/sampling-the-hemisphere/
#[tracked]
fn sample_bounce(normal: Expr<Vec3<f32>>, rng: Expr<Vec2<f32>>) -> Expr<Vec3<f32>> {
    let theta = rng.x.acos();
    let phi = rng.y * TAU;
    let v = Vec3::expr(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    );
    rotate_hemi(normal, v)
}

#[tracked]
fn over(fluence: Expr<Fluence>, radiance: Expr<Radiance>) -> Expr<Radiance> {
    fluence.xyz() + radiance * fluence.w
}

#[tracked]
fn sunlight_radiance(dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
    let sunlight_dir = Vec3::expr(0.3_f32, 1.5, 1.0).normalize();
    if dir.dot(sunlight_dir) > 0.9 {
        Vec3::splat_expr(10.0_f32)
    } else {
        Vec3::splat_expr(0.01_f32)
    }
}

type NoiseTexture = Tex3d<Vec4<f32>>;
fn load_noise_textures(names: impl Fn(u32) -> PathBuf) -> NoiseTexture {
    let storage = PixelStorage::Byte4;
    let mut num_images = 0;
    let mut data = vec![];
    let mut shape = (0, 0);
    while let Ok(image) = ImageReader::open(names(num_images)) {
        let image = image.decode().unwrap();
        if shape == (0, 0) {
            shape = image.dimensions();
        } else {
            assert_eq!(shape, image.dimensions());
        }
        num_images += 1;
        assert_eq!(image.color(), image::ColorType::Rgba8);

        data.extend(image.into_rgba8().pixels().map(|p| Vec4::from(p.0)));
    }
    let texture = DEVICE.create_tex3d(storage, shape.0, shape.1, num_images, 1);
    texture.view(0).copy_from(&data);
    texture
}
fn load_noise2_textures(names: impl Fn(u32) -> PathBuf) -> Tex3d<Vec2<u32>> {
    let storage = PixelStorage::Byte2;
    let mut num_images = 0;
    let mut data = vec![];
    let mut shape = (0, 0);
    while let Ok(image) = ImageReader::open(names(num_images)) {
        let image = image.decode().unwrap();
        if shape == (0, 0) {
            shape = image.dimensions();
        } else {
            assert_eq!(shape, image.dimensions());
        }
        num_images += 1;
        assert_eq!(image.color(), image::ColorType::Rgb8);

        data.extend(
            image
                .into_rgb8()
                .pixels()
                .map(|p| Vec2::new(p.0[0], p.0[1])),
        );
    }
    let texture = DEVICE.create_tex3d(storage, shape.0, shape.1, num_images, 1);
    texture.view(0).copy_from(&data);
    texture
}

#[tracked]
fn as_normal(v: Expr<Vec4<f32>>) -> Expr<Vec3<f32>> {
    v.xyz() * 2.0 - 1.0
}

#[repr(C)]
#[derive(Value, Debug, Clone, Copy)]
struct Ray {
    dir: Vec3<f32>,
    start: Vec3<f32>,
    base_probe: Probe,
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [1920, 1080])
        .scale(2)
        .agx()
        .resize()
        .init();

    yesod::printer::global::set_capacity(1024 * 1024);

    // let sphere_noise =

    let hemi_noise = load_noise_textures(|i| {
        PathBuf::from(format!(
            "assets/noise/sphere/exp/sphere_coshemi_binomial3x3_exp0101_product_{i}.png",
        ))
    });
    let noise2 = load_noise2_textures(|i| PathBuf::from(format!("assets/noise2/{i}.png",)));

    /*let world = {
        let a = 6000.0;
        let b = 150.0;
        let h1 = 7.0;
        let h2 = 307.0;

        TriangleWorld::new(
            &[
                [-a, -a, h1],
                [-a, a, h1],
                [a, -a, h1],
                [a, a, h1],
                [-b, -b, h2],
                [-b, b, h2],
                [-b, -b, h1],
                [-b, b, h1],
            ],
            &[[0.0; 3]; 4],
            &[[0, 2, 1], [1, 2, 3], [0, 2, 1], [1, 2, 3]],
            &[[0, 0], [2, 4]],
        )
    };*/

    let world = TriangleWorld::load(
        "assets/sponza.obj",
        FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    );

    let depth_buf =
        DEVICE.create_tex2d::<f32>(PixelStorage::Float1, app.size()[0], app.size()[1], 1);
    let depth_buf = BINDLESS.push_tex2d(
        depth_buf,
        Sampler {
            filter: SamplerFilter::LinearLinear,
            address: SamplerAddress::Edge,
        },
    );
    BINDLESS.update();
    let normal_buf =
        DEVICE.create_tex2d::<Vec2<f32>>(PixelStorage::Short2, app.size()[0], app.size()[1], 1);
    let accum_buf =
        DEVICE.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, app.size()[0], app.size()[1], 1);
    let radiance_buf = DEVICE.create_buffer::<Vec4<f32>>((app.size()[0] * app.size()[1]) as usize);
    // let cache = HashCache::new(16.0, 100000.0, 4, 1 << 18, 6);
    let cache = HashCache::new(1.0, 100.0, 4, 1 << 17, 6);
    let primary_cache = DEVICE.create_buffer_from_fn::<Vec4<f32>>(
        (cache.base_nodes
            * cache.base_angular_resolution.pow(2)
            * INDEX_SCALING.x
            * INDEX_SCALING.y) as usize,
        |_| Vec4::new(0.0, 0.0, 0.0, 0.00001),
    );

    let base_probe_num_rays =
        DEVICE.create_buffer_from_fn::<u32>(cache.base_nodes as usize, |_| 0_u32);
    let node_quality = DEVICE.create_buffer_from_fn::<f32>(cache.nodes.len(), |_| 0.0_f32);
    let node_lifetime = DEVICE.create_buffer_from_fn::<u32>(cache.nodes.len(), |_| 0_u32);
    let traced_rays = DEVICE.create_buffer::<Ray>(MAX_RAYS as usize);
    let next_traced_ray = Singleton::<u32>::new(0_u32);

    let cache_memory = cache.data.len() * std::mem::size_of::<RadianceElement>()
        + cache.nodes.len() * std::mem::size_of::<Node>();
    println!("Cache memory: {} MB", cache_memory / 1024 / 1024);

    let cache_size = DEVICE.create_buffer::<u32>(cache.num_levels as usize);

    let decay_cache = DEVICE.create_kernel::<fn(u32)>(&track!(|level| {
        let node_index = dispatch_id().x + cache.get_node_offset(level);
        let node = cache.nodes.read(node_index).var();
        if node.position == POSITION_INVALID {
            return;
        }
        let quality = node_quality.read(node_index).var();
        *quality *= QUALITY_DECAY;
        let lifetime = node_lifetime.read(node_index).var();
        *lifetime -= 1_u32;
        let is_invalid = lifetime == 0;
        if is_invalid {
            *node.position = POSITION_INVALID;
            *quality = 0.0;
        }
        cache.nodes.write(node_index, **node);
        node_quality.write(node_index, **quality);
        node_lifetime.write(node_index, **lifetime);

        if is_invalid {
            let position = decode_position(**node.position);
            let probe = Probe::from_comps_expr(ProbeComps { position, level });
            let node = cache.load_index(probe, dispatch_id().x);

            for index in 0_u32.expr()..node.encoder.linear_size() {
                cache
                    .data
                    .write(node.data_offset + index, RadianceElement::default());
            }
            if level == 0 {
                for index in 0_u32.expr()..node.encoder.linear_size() {
                    primary_cache
                        .write(node.data_offset + index, Vec4::new(0.0, 0.0, 0.0, 0.00001));
                }
            }
        } else {
            cache_size.atomic_ref(level).fetch_add(1_u32.expr());
        }
    }));
    let reset_tex2d = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>)>(&track!(|tex| {
        let pixel = dispatch_id().xy();
        tex.write(pixel, Vec4::splat_expr(0.0_f32));
    }));
    let reset_cache_nodes = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache.nodes.write(dispatch_id().x, Node::default());
        node_quality.write(dispatch_id().x, 0.0_f32);
    }));
    let reset_cache_data = DEVICE.create_kernel::<fn()>(&track!(|| {
        cache
            .data
            .write(dispatch_id().x, RadianceElement::default());
    }));
    let reset_primary_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        primary_cache.write(dispatch_id().x, Vec4::new(0.0, 0.0, 0.0, 0.00001));
    }));

    let reset_base_probe_num_rays = DEVICE.create_kernel::<fn()>(&track!(|| {
        base_probe_num_rays.write(dispatch_id().x, 0_u32);
    }));
    let reset_next_traced_ray = DEVICE.create_kernel::<fn()>(&track!(|| {
        next_traced_ray.write(0_u32);
    }));

    let reset_radiance_buf = DEVICE.create_kernel::<fn()>(&track!(|| {
        radiance_buf.write(dispatch_id().x, Vec4::splat_expr(0.0_f32));
    }));

    let get_fluence = track!(|start: Expr<Vec3<f32>>,
                              dir: Expr<Vec3<f32>>,
                              hit: Expr<SurfaceHit>| {
        if hit.valid() {
            let bounce_pos = start + dir * hit.committed_ray_t;
            let normal = world.get_normal_flat(hit);
            let weight = normal.dot(sunlight_dir());
            let radiance = Vec3::splat(0.0_f32).var();

            if weight > 0.0 {
                *radiance += world
                    .trace_segment(
                        bounce_pos,
                        sunlight_dir().expr(),
                        Vec2::expr(0.01, 100000.0),
                    )
                    .valid()
                    .select(Vec3::splat_expr(0.0_f32), Vec3::splat_expr(10.0_f32))
                    * weight;
            }
            if hit.inst == 1 {
                *radiance += Vec3::expr(10.0, 0.0, 0.0);
            }
            radiance.extend(0.0)
        } else {
            Vec3::splat_expr(0.0_f32).extend(1.0)
        }
    });

    #[derive(Clone, Copy)]
    enum MergeLevel {
        // First,
        Last,
        Middle(Expr<u32>),
    }
    // TODO: Can reduce memory by adding 2x2 blocks. Morton-code for further efficiencies.
    let merge_cache_base = track!(|merge_level: MergeLevel| {
        set_block_size([1, 32, 1]);
        let level = match merge_level {
            // MergeLevel::First => 0.expr(),
            MergeLevel::Last => (cache.num_levels - 1).expr(),
            MergeLevel::Middle(l) => l,
        };
        let node_index = dispatch_id().x;
        let node = cache.read_node(level, node_index);
        if node.position == POSITION_INVALID {
            return;
        }
        let position = decode_position(node.position);
        let probe = Probe::from_comps_expr(ProbeComps { position, level });
        let node = cache.load_index(probe, node_index);
        // TODO: best to process in batches of 8.
        // TODO: pre-load all of the data batched.
        let next_probes = (!matches!(merge_level, MergeLevel::Last)).then(|| {
            probe.next_bilinear().map(|(probe, weight)| {
                let (next, weight) = cache.query(
                    probe,
                    |n| (n, weight),
                    || {
                        (
                            LoadedNode::from_comps_expr(LoadedNodeComps {
                                probe,
                                encoder: cache.get_encoder(probe.level),
                                data_offset: 0_u32.expr(),
                                index: 0_u32.expr(),
                            }),
                            0_f32.expr(),
                        )
                    },
                );
                (next, weight)
            })
        });
        for index in 0_u32.expr()..(node.encoder.linear_size() / dispatch_size().y) {
            let index = index * dispatch_size().y + dispatch_id().y;
            let dir = node.encoder.linear_to_grid(index);
            let total_radiance = Vec4::splat(0.0_f32).var();
            if let MergeLevel::Last = merge_level {
                // Shouldn't be used.
                *total_radiance = Vec4::expr(10000.0, 0.0, 0.0, 1.0);
            } else {
                for offset in [[0, 0], [0, 1], [1, 0], [1, 1]].map(Vec2::from) {
                    let sample_dir = dir * 2 + offset;
                    // TODO: Can optimize because half are empty.
                    for (next, weight) in next_probes.unwrap() {
                        let sample_index = next.encoder.grid_to_linear(sample_dir);
                        let radiance = cache.data.read(next.data_offset + sample_index).radiance();
                        if radiance.is_finite().all() {
                            *total_radiance += weight * radiance.extend(1.0);
                        }
                    }
                }
            }
            let data_index = node.data_offset + index;

            let far = total_radiance.xyz() / total_radiance.w;

            let radiance_deposit = cache.data.read(data_index).var();

            /*{
                let probe_pos = cache.dequantize(probe);
                let segment = (12.0
                    * Vec2::new(0.5, 1.0)
                    * cache.base_grid_size
                    * (1_u32 << (probe.position.w + probe.level)).cast_f32())
                .var();
                if matches!(merge_level, MergeLevel::Last) {
                    *segment.y = 999999.0;
                }
                if level == 0 {
                    *segment.x =
                        2.0 * cache.base_grid_size * (1_u32 << probe.position.w).cast_f32();
                }
                let dir = node.encoder.linear_to_dir(index);
                let hit = world.trace_segment(probe_pos, dir, **segment);
                let fluence = get_fluence(probe_pos, dir, hit).var();
                if matches!(merge_level, MergeLevel::Last) {
                    *fluence.w = 0.0;
                }
                let radiance = fluence.xyz() + fluence.w * far;
                *radiance_deposit.radiance_weight.x = radiance.x;
                *radiance_deposit.radiance_weight.y = radiance.y;
                *radiance_deposit.radiance_weight.z = radiance.z;
            }*/

            radiance_deposit.merge(far);
            cache.data.write(data_index, radiance_deposit);
        }
    });

    let add_primary_cache = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let node = cache.read_node(0.expr(), index);
        if node.position == POSITION_INVALID {
            return;
        }
        let node = cache.load_index(
            Probe::from_comps_expr(ProbeComps {
                position: decode_position(node.position),
                level: 0.expr(),
            }),
            index,
        );
        for index in 0_u32.expr()..node.encoder.linear_size() {
            let data_index = node.data_offset + index;
            let radiance_deposit = cache.data.read(data_index).var();
            let primary = primary_cache.read(data_index);
            *radiance_deposit.radiance_weight += (primary.xyz() / primary.w).extend(0.0);
            cache.data.write(data_index, radiance_deposit);
        }
    }));

    let merge_cache = DEVICE.create_kernel::<fn(u32)>(&|level| {
        merge_cache_base(MergeLevel::Middle(level));
    });
    let merge_last_cache = DEVICE.create_kernel::<fn()>(&|| {
        merge_cache_base(MergeLevel::Last);
    });
    let compute_gbuffer_kernel = DEVICE.create_kernel::<fn(View)>(&track!(|view| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let hit = world.trace(view.pos, ray_dir);
        if !hit.valid() {
            depth_buf.write(pixel, 0.0);
            return;
        }
        depth_buf.write(pixel, hit.committed_ray_t);
        let normal = world.get_normal_flat(hit);
        normal_buf.write(pixel, SphericalEncoder::encode(normal));
    }));

    let compute_num_rays_kernel = DEVICE.create_kernel::<fn(View, u32)>(&track!(|view, _tick| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let depth = depth_buf.read(pixel);
        if depth == 0.0 {
            return;
        }
        let world_pos = (view.pos + ray_dir * depth).var();

        let interpolate_side = pixel.x % 2 == pixel.y % 2;

        let probe = cache.quantize(**world_pos, interpolate_side);

        let node = cache.query_init(
            probe,
            |n| n,
            || {
                return;
                **LoadedNode::var_zeroed()
            },
        );
        base_probe_num_rays
            .atomic_ref(node.index)
            .fetch_add(1_u32.expr());
    }));
    let compute_rays_kernel = DEVICE.create_kernel::<fn(View, u32)>(&track!(|view, tick| {
        set_block_size([8, 8, 1]);
        let pixel = dispatch_id().xy();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

        let depth = depth_buf.read(pixel);
        if depth == 0.0 {
            return;
        }
        let normal = SphericalEncoder::decode(normal_buf.read(pixel));

        let world_pos = view.pos + ray_dir * depth;

        let interpolate_side = pixel.x % 2 == pixel.y % 2;

        let probe = cache.quantize(world_pos, interpolate_side);

        let node = cache.query_init(
            probe,
            |n| n,
            || {
                return;
                **LoadedNode::var_zeroed()
            },
        );

        /*let bounce_ray = noise2
            .read(pixel.extend(tick) % Vec3::new(noise2.width(), noise2.height(), noise2.depth()))
            / 2;
        let bounce_ray = bounce_ray.cast_f32() + 0.5;
        let bounce_ray = bounce_ray / 128.0;
        let bounce_ray = bounce_ray * Vec2::new(0.5, 1.0); // X-axis is doubled because only hemisphere / mirroring.
        let bounce_ray = SphericalEncoder::decode(bounce_ray);

        let bounce_ray = bounce_ray * bounce_ray.dot(normal).signum();
        */
        let bounce_ray = rotate_hemi(
            normal,
            as_normal(hemi_noise.read(
                pixel.extend(tick)
                    % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
            )),
        );

        let num_rays = base_probe_num_rays.read(node.index);
        let target_rays = keter::max(
            cache.get_encoder(0.expr()).linear_size().cast_f32() * TARGET_QUALITY_MULTIPLIER
                - node_quality.read(node.index),
            0.0,
        );
        let extinction = target_rays / keter::max(num_rays, 1).cast_f32();

        if pcg3df((pixel + Vec2::new(3528, 9306)).extend(tick)).x < extinction {
            let ray = Ray::from_comps_expr(RayComps {
                dir: bounce_ray,
                start: world_pos,
                base_probe: probe,
            });
            let ray_index = next_traced_ray.atomic().fetch_add(1_u32.expr());
            if ray_index < traced_rays.len_expr_u32() {
                traced_rays.write(ray_index, ray);
            }
        }
    }));

    let compute_radiance_kernel = DEVICE.create_kernel_with_options::<fn(u32, u32, f32)>(
        KernelBuildOptions {
            name: Some("compute_radiance_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|_tick, flags, c0_len| {
            set_block_size([64, 1, 1]);

            if dispatch_id().x >= next_traced_ray.read() {
                return;
            }

            let ray = traced_rays.read(dispatch_id().x);
            let ray_dir = ray.dir.normalize();
            let world_pos = ray.start;

            let bounce_ray = ray_dir;

            let probe = ray.base_probe;

            let bounce_hit = world.trace_segment(
                world_pos,
                bounce_ray,
                Vec2::expr(0.01, 100000.0), // * (1_u32 << probe.position.w).cast_f32(),
            );

            let (hit_level, fluence) = if bounce_hit.valid() {
                let level = keter::max(
                    (bounce_hit.committed_ray_t
                        / (c0_len * cache.base_grid_size * (1_u32 << probe.position.w).cast_f32()))
                    .log2(),
                    0.0,
                )
                .ceil()
                .cast_u32();
                (
                    keter::min(level, cache.num_levels.expr() - 1),
                    get_fluence(world_pos, bounce_ray, bounce_hit),
                )
            } else {
                (cache.num_levels.expr() - 1, Vec4::splat_expr(0.0))
            };

            let deposit = |level, fluence| {
                cache.query_init(
                    probe.shift_level(level),
                    |node| {
                        let data = cache
                            .data
                            .atomic_ref(node.data_offset + node.encoder.dir_to_linear(bounce_ray));
                        data.deposit(RadianceDeposit::expr(fluence));
                        node_quality.atomic_ref(node.index).fetch_add(1.0);
                        node_lifetime
                            .atomic_ref(node.index)
                            .fetch_max(NODE_LIFETIME.expr());
                    },
                    || {},
                )
                // cache.deposit(probe.shift_level(level), bounce_ray, RadianceDeposit::expr(fluence));
            };

            if (flags & TERMINATE_DEPOSIT) == 0 {
                for level in 0..cache.num_levels {
                    deposit(
                        level,
                        if level < hit_level {
                            Vec3::splat_expr(0.0_f32).extend(1.0)
                        } else {
                            fluence
                        },
                    );
                }
            } else {
                for level in 0.expr()..hit_level {
                    deposit(level, Vec3::splat_expr(0.0_f32).extend(1.0));
                }
                deposit(hit_level, fluence);
            }
        }),
    );

    let compute_screen_kernel =
        DEVICE.create_kernel::<fn(View, f32)>(&track!(|view, blend_radius| {
            set_block_size([1, 32, 1]);
            let node = cache.nodes.read(dispatch_id().x);
            if node.position == POSITION_INVALID
                || node_lifetime.read(dispatch_id().x) != NODE_LIFETIME
            {
                return;
            }
            let position = decode_position(node.position);
            let probe = Probe::from_comps_expr(ProbeComps {
                position,
                level: 0.expr(),
            });
            let node = cache.load_index(probe, dispatch_id().x);
            let quality = node_quality.read(dispatch_id().x);

            let radiance_data = Shared::<Vec3<f32>>::new(32);
            let v = cache
                .data
                .read(node.data_offset + dispatch_id().y)
                .radiance();
            radiance_data.write(dispatch_id().y, v);
            sync_block();

            let screen_min = Vec2::from(app.size()).expr().cast_f32().var();
            let screen_max = Vec2::splat(0.0_f32).var();

            let spacing = cache.base_grid_size * (1 << probe.position.w).cast_f32() * blend_radius;
            let probe_pos = cache.dequantize(probe);

            // TODO: May go schizo if the probe is behind the camera...
            for i in 0..8 {
                let coord = probe_pos
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
            let screen_min = keter::max(0.0_f32, **screen_min).floor().cast_u32();
            let screen_max = keter::min(Vec2::from(app.size()).expr().cast_f32(), **screen_max)
                .ceil()
                .cast_u32();
            let size = screen_max - screen_min;
            let area = size.reduce_prod();
            let i = dispatch_id().y.var();
            while i < area {
                let pixel = screen_min + Vec2::expr(i % size.x, i / size.x);
                *i += dispatch_size().y;

                let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();
                let depth = depth_buf.read(pixel);
                if depth == 0.0 {
                    continue;
                }
                let world_pos = view.pos + ray_dir * depth;
                let normal = SphericalEncoder::decode(normal_buf.read(pixel));
                let weight = keter::max(1.0 - (world_pos - probe_pos).abs() / spacing, 0.0)
                    .reduce_prod()
                    * keter::min(quality, TARGET_QUALITY_MULTIPLIER * 32.0 * 0.75)
                    * (1 << (2 * probe.position.w)).cast_f32().recip();
                if weight == 0.0 {
                    continue;
                }
                let (factor, cascade) = cache.projection().interpolation_factor_cascade(world_pos);
                let factor = if cascade == probe.position.w {
                    factor
                } else {
                    1.0 - factor
                };
                let weight = weight * factor;

                let total_radiance = Vec4::splat(0.0_f32).var();
                for j in 0..32_u32 {
                    let dir = node.encoder.linear_to_dir(j);
                    let weight = dir.dot(normal);
                    if weight > 0.0 {
                        let radiance = radiance_data.read(j);
                        if radiance.is_finite().all() {
                            *total_radiance += weight * radiance_data.read(j).extend(1.0);
                        }
                    }
                }
                // TODO: Figure out the biasing properly? Just directly bias based on amount of missing data?
                if total_radiance.w < 0.0001 {
                    continue;
                }
                let total_radiance = total_radiance.xyz() / total_radiance.w;
                let total_radiance = total_radiance.extend(1.0) * weight;
                let dest = radiance_buf.atomic_ref(pixel.y * app.size()[0] + pixel.x);
                dest.x.fetch_add(total_radiance.x);
                dest.y.fetch_add(total_radiance.y);
                dest.z.fetch_add(total_radiance.z);
                dest.w.fetch_add(total_radiance.w);
            }
        }));

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32, u32, u32, Vec2<u32>)>(
        KernelBuildOptions {
            name: Some("draw_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, _flags, mode, level, cursor_pixel| {
            set_block_size([8, 8, 1]);

            let pixel = dispatch_id().xy();
            let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

            let depth = depth_buf.read(pixel);
            if depth == 0.0 {
                app.set_pixel(pixel.cast_i32(), sunlight_radiance(ray_dir));
                return;
            }
            let normal = SphericalEncoder::decode(normal_buf.read(pixel));
            let world_pos = view.pos + ray_dir * depth;

            let probe = cache.quantize(world_pos, false.expr()).shift_level(level);

            let sunlight = world
                .trace_segment(world_pos, sunlight_dir().expr(), Vec2::expr(0.01, 100000.0))
                .valid()
                .select(Vec3::splat_expr(0.0_f32), Vec3::splat_expr(10.0_f32));

            let color = if mode == 0 {
                let r = radiance_buf.read(pixel.y * app.size()[0] + pixel.x);
                let result = r.xyz() / r.w;
                result + sunlight * keter::max(normal.dot(sunlight_dir()), 0.0)
            } else if mode == 1 {
                if cache.contains(probe) {
                    inferno((probe.position.w.cast_f32() + 0.5) / 9.0)
                        * pcg3df(probe.position.xyz()).x
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else if mode == 2 {
                cache.query(
                    probe,
                    |node| {
                        let quality = node_quality.read(node.index);
                        let quality_ratio = quality
                            / (TARGET_QUALITY_MULTIPLIER
                                * cache.get_encoder(level).linear_size().cast_f32());
                        if (cursor_pixel == pixel).all() {
                            device_println!(
                                "Probe at quality: {}",
                                // probe.position.w.host(),
                                host!(quality)
                            );
                        }
                        inferno(quality_ratio)
                    },
                    || Vec3::splat_expr(0.0),
                )
            } else if mode == 3 {
                let probe = cache.quantize(world_pos, false.expr());
                cache.query(
                    probe,
                    |node| {
                        let num_rays = base_probe_num_rays.read(node.index);
                        let target_rays = keter::max(
                            cache.get_encoder(0.expr()).linear_size().cast_f32()
                                * TARGET_QUALITY_MULTIPLIER
                                - node_quality.read(node.index),
                            0.0,
                        );
                        let extinction = target_rays / keter::max(num_rays, 1).cast_f32();
                        inferno(extinction)
                    },
                    || Vec3::splat_expr(0.0),
                )
                // inferno(1.0 - reinhard(depth / 1000.0))
            } else if mode == 4 {
                // let offset = pcg3d(Vec3::expr(tick / hemi_noise.depth(), 392, 132)).xy()
                //     % Vec2::new(hemi_noise.width(), hemi_noise.height());
                // let bounce_ray = as_normal(hemi_noise.read(
                //     (pixel + offset).extend(tick)
                //         % Vec3::new(hemi_noise.width(), hemi_noise.height(), hemi_noise.depth()),
                // ));
                // let bounce_ray = rotate_hemi(normal, bounce_ray);
                let bounce_ray = sample_bounce(normal, pcg3df(pixel.extend(tick)).xy());

                let bounce_hit =
                    world.trace_segment(world_pos, bounce_ray, Vec2::expr(0.1_f32, 10000.0));
                let radiance = get_fluence(world_pos, bounce_ray, bounce_hit).xyz();
                let value = accum_buf.read(pixel) + radiance.extend(1.0);
                accum_buf.write(pixel, value);
                value.xyz() / value.w + sunlight * keter::max(normal.dot(sunlight_dir()), 0.0)
            } else if mode == 5 {
                inferno(1.0 - reinhard((cache.dequantize(probe) - world_pos).length()))
            } else if mode == 6 {
                let hit = world.trace(view.pos, ray_dir);
                if (cursor_pixel == pixel).all() {
                    let inst = hit.inst;
                    device_println!("Hit at {}", host!(inst));
                }
                pcg3df(Vec3::expr(hit.inst, 37, 926))
            } else if mode == 7 {
                let (has_probe, cost) = cache.contains_cost(probe);
                if has_probe {
                    if cost > 8 {
                        Vec3::expr(1.0, 1.0, 1.0)
                    } else {
                        plasma(cost.cast_f32() / 9.0)
                    }
                } else {
                    Vec3::splat_expr(0.0)
                }
            } else {
                Vec3::expr(1.0, 0.0, 0.0)
            };
            app.set_pixel(pixel.cast_i32(), color);
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        // yaw: 6.857436,
        // pitch: 0.12587574,
        // pos: glam::Vec3::new(308.5938, -573.01685, 240.64691),
        yaw: 6.437008,
        pitch: 0.47492924,
        pos: glam::Vec3::new(-94.45221, -937.2442, 613.4341),
        fov: 1.15,
    };
    let mut running = true;
    let mut mode = 0;
    let mut level = 0;
    let mut radiance_flags = TERMINATE_DEPOSIT;
    let mut c0_len_mode = 0;
    let mut blend_radius_mode = 0;

    let mut profiler = Profiler::new();

    let v = vec![0_u32; cache.num_levels as usize];

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            20.0
        } else {
            800.0
        } * rt.frame_time();
        let camera_updated = camera.apply_movement(&rt.keys_down, rot_speed, move_speed);
        if rt.key_pressed(KeyCode::KeyP) {
            println!("Camera: {camera:?}");
        }
        if rt.key_pressed(KeyCode::KeyM) {
            mode = (mode + 1) % 8;
            println!("Drawing mode {mode}");
            let tonemap = matches!(mode, 0 | 4);
            rt.tonemap(tonemap);
        }
        if rt.key_pressed(KeyCode::KeyL) {
            level = (level + 1) % cache.num_levels;
            println!("Level set {level}");
        }
        if rt.key_pressed(KeyCode::KeyK) {
            c0_len_mode = (c0_len_mode + 1) % 3;
        }
        if rt.key_pressed(KeyCode::KeyO) {
            running = !running;
            println!("Running: {running}");
        }
        if rt.key_pressed(KeyCode::KeyT) {
            radiance_flags ^= TERMINATE_DEPOSIT;
        }
        if rt.key_pressed(KeyCode::KeyB) {
            blend_radius_mode = (blend_radius_mode + 1) % 3;
        }
        let mut commands = vec![];
        commands.push(
            compute_gbuffer_kernel
                .dispatch_async(rt.dispatch_size(), &camera.view())
                .debug("Compute GBuffer"),
        );

        let running = running || rt.key_pressed(KeyCode::Period);
        if running {
            if mode == 3 {
                println!("Num rays traced: {}", next_traced_ray.read_blocking());
            }
            commands.push(cache.set_view(camera.view()));

            commands.push(cache_size.copy_from_async(&v).debug("Copy Cache Size"));

            for l in (0..cache.num_levels).rev() {
                let dispatch_size = [cache.num_nodes_at_level(l), 1, 1];
                commands.push(
                    decay_cache
                        .dispatch_async(dispatch_size, &l)
                        .debug("Decay Cache"),
                );
            }

            // Reset ray counts
            commands.push(
                reset_base_probe_num_rays
                    .dispatch_async([base_probe_num_rays.len() as u32, 1, 1])
                    .debug("Reset Base Probe Num Rays"),
            );
            commands.push(
                reset_next_traced_ray
                    .dispatch_async([1, 1, 1])
                    .debug("Reset Next Traced Ray"),
            );

            commands.push(
                compute_num_rays_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick)
                    .debug("Compute Radiance Num Rays"),
            );
            commands.push(
                compute_rays_kernel
                    .dispatch_async(rt.dispatch_size(), &camera.view(), &rt.tick)
                    .debug("Compute Radiance Rays"),
            );
            commands.push(
                compute_radiance_kernel
                    .dispatch_async(
                        [MAX_RAYS, 1, 1],
                        &rt.tick,
                        &radiance_flags,
                        &match c0_len_mode {
                            0 => 4.0,
                            1 => 12.0,
                            2 => 999.0,
                            _ => unreachable!(),
                        },
                    )
                    .debug("Compute Radiance Trace"),
            );

            for l in (0..cache.num_levels).rev() {
                let dispatch_size = [cache.num_nodes_at_level(l), 1 << l, 1];
                commands.push(
                    (if l == cache.num_levels - 1 {
                        merge_last_cache.dispatch_async(dispatch_size)
                    } else {
                        merge_cache.dispatch_async(dispatch_size, &l)
                    })
                    .debug(format!("Merge Cache Level {l}")),
                );
            }
        }

        if camera_updated {
            commands.push(
                reset_tex2d
                    .dispatch_async([accum_buf.width(), accum_buf.height(), 1], &accum_buf)
                    .debug("Reset Accumulation"),
            );
        }

        commands.push(
            reset_radiance_buf
                .dispatch_async([radiance_buf.len() as u32, 1, 1])
                .debug("Reset Radiance Buffer"),
        );
        commands.push(
            compute_screen_kernel
                .dispatch_async(
                    [cache.base_nodes, 32, 1],
                    &camera.view(),
                    &match blend_radius_mode {
                        0 => 1.0,
                        1 => 0.5,
                        2 => 0.6,
                        _ => unreachable!(),
                    },
                )
                .debug("Compute Screen"),
        );

        commands.push(
            draw_kernel
                .dispatch_async(
                    rt.dispatch_size(),
                    &camera.view(),
                    &rt.tick,
                    &radiance_flags,
                    &mode,
                    &level,
                    &rt.cursor_position.map(|x| x as u32),
                )
                .debug("Draw"),
        );

        if running {
            commands.push(
                add_primary_cache
                    .dispatch_async([cache.num_nodes_at_level(0), 1, 1])
                    .debug("Add Primary Cache"),
            );
        }

        profiler.record(commands.chain().execute_timed());

        let cache_size = cache_size.copy_to_vec();

        if profiler.time() > 1000.0 {
            profiler.print(
                &[
                    "Compute GBuffer",
                    "Decay Cache",
                    "Compute Radiance",
                    "Merge",
                    "Add Primary Cache",
                    "Compute Screen",
                    "Draw",
                ],
                true,
            );
            println!("Cache Sizes per Level: {:?}", cache_size);
            println!(
                "Max: {:?}",
                (0..cache.num_levels)
                    .map(|l| cache.num_nodes_at_level(l))
                    .collect::<Vec<_>>()
            );
            profiler.reset();
        }

        if rt.key_down(KeyCode::Backslash) {
            // println!("Resetting cache");
            reset_cache_nodes.dispatch([cache.nodes.len() as u32, 1, 1]);
            reset_cache_data.dispatch([cache.data.len() as u32, 1, 1]);
            reset_primary_cache.dispatch([primary_cache.len() as u32, 1, 1]);
        }
        flush_printer();
    });
}
