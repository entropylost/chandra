/*
TODOS:
[ ] Better hashmap impl without dropping things? Bias towards primary rays first.
[ ] Some sorta anderson acceleration to make the infinite bounce convergence less obvious?

[ ] Precompute ndotl lighting separately? At least for offscreen?
[ ] Add separate world-space cache with lower resolution? Preferably get both to use the same map to prevent overflow...
[ ] Ray counting / quality metrics.
[ ] Test how much memory saved if each probe element is sparse?

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

use std::f32::consts::PI;
use std::path::Path;

use chandra::algorithm::ddgi::{DdgiConfig, DdgiData};
use chandra::algorithm::rc::{RcConfig, RcData};
use chandra::algorithm::splat::SplatData;
use chandra::algorithm::trace::TraceData;
use chandra::algorithm::{CommandList, CoreData, WorldData};
use chandra::cache::{DirectionIndexer, DirectionIndexerComps};
use chandra::utils::sample_bounce;
use chandra::world::TriangleWorld;
use glam::Mat3 as FMat3;
use image::Rgb32FImage;
use keter::graph::profile::Profiler;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};
use yesod::camera::{Camera, View};
use yesod::color::inferno;
use yesod::printer::global::flush_printer;
use yesod::rand::{pcg3df, r2};
use yesod::utils::{dispatch0, offset_ray};
use yesod::{device_println, host};

const TARGET_QUALITY_MULTIPLIER: f32 = 10.0;
const DEBUG_SPLATS: u32 = 0b1;
const DRAW_SPECULAR: u32 = 0b10;
const DRAW_DIFFUSE: u32 = 0b100;

fn load_envmap(path: impl AsRef<Path>) -> Rgb32FImage {
    image::open(path).unwrap().to_rgb32f()
}

struct Scene {
    world: WorldData,
    rc_config: RcConfig,
    ddgi_bounds: (glam::Vec3, glam::Vec3),
    ddgi_probe_dims: glam::UVec3,

    camera_yaw: f32,
    camera_pitch: f32,
    camera_pos: glam::Vec3,
}
impl Scene {
    fn sponza_main() -> Self {
        Self {
            world: WorldData {
                world: TriangleWorld::load(
                    "assets/sponza.obj",
                    FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
                    Some(Vec3::splat(0.7)),
                ),
                envmap: load_envmap("assets/qwantani_dusk_2_4k.exr"),
                envmap_scale: 1.0,
                sun_dir: glam::Vec3::new(0.1, 0.5, 1.0).normalize().into(),
                sun_color: Vec3::splat(10.0),
                custom_emitters: vec![
                    (1, Vec3::new(10.0, 0.0, 0.0)),
                    (9, Vec3::new(0.0, 7.0, 7.0)),
                ],
            },
            rc_config: RcConfig {
                // base_grid_size: 11.0,
                // base_probes: 1 << 16,
                base_grid_size: 6.0,
                base_probes: 1 << 17,
                ..Default::default()
            },
            ddgi_bounds: (
                glam::Vec3::new(-700.0, -1600.0, -10.0),
                glam::Vec3::new(700.0, 1300.0, 900.0),
            ),
            ddgi_probe_dims: glam::UVec3::new(32, 32, 16),

            camera_yaw: 6.437008,
            camera_pitch: 0.47492924,
            camera_pos: glam::Vec3::new(-94.45221, -937.2442, 613.4341),
        }
    }
    fn sponza_penumbra() -> Self {
        Self {
            camera_yaw: 9.474376,
            camera_pitch: 1.1788585,
            camera_pos: glam::Vec3::new(-61.073547, 1171.4944, 375.83527),
            ..Self::sponza_main()
        }
    }
    fn sanmiguel_teaser() -> Self {
        Self {
            world: WorldData {
                world: TriangleWorld::load(
                    "assets/san-miguel-colored.obj",
                    FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 100.0,
                    None,
                ),
                envmap: load_envmap("assets/qwantani_dusk_2_4k.exr"),
                envmap_scale: 0.5,
                sun_dir: glam::Vec3::new(0.1, 2.0, 1.0).normalize().into(),
                sun_color: Vec3::splat(30.0),
                custom_emitters: vec![],
            },
            rc_config: RcConfig {
                // base_grid_size: 12.0,
                // base_probes: 1 << 16,
                base_grid_size: 8.0,
                base_probes: 1 << 18,
                use_quality_weight: true,
                blending_radius: 1.3,
                ..Default::default()
            },
            ddgi_bounds: (
                glam::Vec3::new(-600.0, -300.0, -100.0),
                glam::Vec3::new(1500.0, 3000.0, 1281.0),
            ),
            ddgi_probe_dims: glam::UVec3::new(32, 48, 32),
            camera_yaw: 10.387859,
            camera_pitch: 0.70613384,
            camera_pos: glam::Vec3::new(351.25754, 2183.237, 802.5459),
        }
    }
    fn sanmiguel_alt() -> Self {
        Self {
            world: WorldData {
                world: TriangleWorld::load(
                    "assets/san-miguel-colored.obj",
                    FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 100.0,
                    None,
                ),
                envmap: load_envmap("assets/qwantani_dusk_2_4k.exr"),
                envmap_scale: 1.0,
                sun_dir: glam::Vec3::new(0.1, 0.6, 1.0).normalize().into(),
                sun_color: Vec3::splat(10.0),
                custom_emitters: vec![],
            },
            rc_config: RcConfig {
                // base_grid_size: 8.0,
                // base_probes: 1 << 16,
                base_grid_size: 4.0,
                base_probes: 1 << 18,
                ..Default::default()
            },
            ddgi_bounds: (
                glam::Vec3::new(-600.0, -300.0, -100.0),
                glam::Vec3::new(1500.0, 3000.0, 1281.0),
            ),
            ddgi_probe_dims: glam::UVec3::new(32, 48, 32),
            camera_yaw: 4.182359,
            camera_pitch: 0.36102834,
            camera_pos: glam::Vec3::new(136.32507, 1907.1962, 244.39528),
        }
    }
    fn conference() -> Self {
        Self {
            world: WorldData {
                world: TriangleWorld::load(
                    "assets/conference.obj",
                    FMat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
                    None,
                ),
                envmap: load_envmap("assets/qwantani_dusk_2_4k.exr"),
                envmap_scale: 0.0,
                sun_dir: glam::Vec3::new(0.1, 0.6, 1.0).normalize().into(),
                sun_color: Vec3::splat(0.0),
                custom_emitters: vec![],
            },
            rc_config: RcConfig {
                base_grid_size: 4.0,
                base_probes: 1 << 19,
                ..Default::default()
            },
            ddgi_bounds: (
                glam::Vec3::new(-600.0, -300.0, -100.0),
                glam::Vec3::new(1500.0, 3000.0, 1281.0),
            ),
            ddgi_probe_dims: glam::UVec3::new(32, 48, 32),

            camera_yaw: 2.3309836,
            camera_pitch: 0.3910516,
            camera_pos: glam::Vec3::new(-811.98444, 1273.2661, 560.3888),
        }
    }
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Chandra", [1920, 1080])
        .scale(1)
        .agx()
        .resize()
        .init();

    yesod::printer::global::set_capacity(1024 * 1024);

    let scene = Scene::sanmiguel_alt();

    let core = CoreData::new(app.size(), scene.world);
    let mut rc = RcData::new(scene.rc_config);
    let trace = TraceData::new(&core, &rc, 1920 * 1080 * 3);
    let splat = SplatData::new(&core, &rc, 1.0);
    // let ris = RisData::new(&core);

    let ddgi = {
        let min = scene.ddgi_bounds.0;
        let max = scene.ddgi_bounds.1;
        let size = max - min;
        let probe_dims = scene.ddgi_probe_dims.into();
        let spacing = size / (glam::UVec3::from(probe_dims).as_vec3() - 1.0);
        let spacing = spacing.max_element();

        DdgiData::new(DdgiConfig {
            spacing: Vec3::splat(spacing),
            origin: min.into(),
            rays_per_probe: 2679,
            hysteresis: 0.97,
            probe_dims,
            ..Default::default()
        })
    };

    #[cfg(feature = "denoise")]
    let mut denoiser = chandra::denoise::Denoiser::new(&core);

    let num_cascades = rc.num_cascades;

    let cache_memory = rc.cache.weighted_fluence.size_bytes()
        + rc.cache.weights.size_bytes()
        + rc.cache.table.memory_size();
    println!("Cache memory: {} MB", cache_memory / 1024 / 1024);

    let accum_buf = DEVICE.create_tex2d::<Vec4<f32>>(
        PixelStorage::Float4,
        core.screen_size().x,
        core.screen_size().y,
        1,
    );

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32, u32, u32, u32, Vec2<u32>)>(
        KernelBuildOptions {
            name: Some("draw_kernel".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, tick, flags, mode, cascade, cursor_pixel| {
            set_block_size([8, 8, 1]);

            let pixel = dispatch_id().xy();
            let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

            let depth = core.read_depth(pixel);
            if depth == 0.0 {
                app.set_pixel(pixel.cast_i32(), core.get_environment(ray_dir));
                return;
            }
            let world_pos = view.pos + ray_dir * depth;
            let normal = core.read_normal(pixel);

            let lod = rc.cache.estimate_lod(world_pos);
            let probe = rc.cache.quantize(world_pos, lod, cascade);

            let pixel_index = core.pixel_to_linear(pixel);
            let hit = core.hit.read(pixel_index);
            let emission = core.get_emission(None, world_pos, hit, ray_dir);
            let diffuse = core.world.get_diffuse(hit);

            let color = if mode == 0 {
                let result = Vec3::splat(0.0f32).var();
                if flags & DRAW_DIFFUSE != 0 {
                    *result = splat.get_radiance(pixel_index) * diffuse;
                }

                if flags & DRAW_SPECULAR != 0 {
                    let specular_dir = chandra::utils::reflect(ray_dir, normal);
                    let specular_radiance = rc.sample(world_pos, 3.expr(), specular_dir).var();
                    for i in (0..3).rev() {
                        let fluence = rc.cache.sample(world_pos, i.expr(), specular_dir);
                        *specular_radiance =
                            chandra::algorithm::rc::over(fluence, **specular_radiance);
                    }
                    if specular_radiance.is_finite().all() {
                        *result += specular_radiance;
                    }
                }

                if flags & DEBUG_SPLATS == 0 {
                    **result + emission
                } else {
                    let radiance_depth = splat.depth.read(pixel_index);
                    if radiance_depth > depth + 0.1 {
                        pcg3df(probe.position.xyz()) * 0.1 + result * 0.9
                    } else {
                        **result
                    }
                }
            } else if mode == 1 {
                // rc.sample_secondary(world_pos, normal)
                let node = rc.cache.get(probe);
                if node.loaded() {
                    let encoder = rc.cache.get_encoder(cascade);
                    let total_transmittance = Vec2::splat(0.0f32).var();
                    for j in 0_u32.expr()..encoder.linear_size() {
                        let transmittance = rc.cache.weighted_fluence.read(node.data_offset + j).w;
                        let weight = rc.cache.weights.read(node.data_offset + j);
                        *total_transmittance += Vec2::expr(transmittance, weight);
                    }
                    let transmittance = total_transmittance.x / total_transmittance.y;
                    inferno(transmittance)
                } else {
                    Vec3::expr(1.0, 0.0, 0.0)
                }
            } else if mode == 2 {
                let output = inferno(yesod::color::reinhard(core.read_depth(pixel) / 1000.0));
                #[cfg(feature = "denoise")]
                let output = denoiser.output.read(pixel_index) * diffuse + emission;
                output
                /*let num_next = 0_f32.var();
                for i in 0..8u32 {
                    let (next_probe, weight) = probe.next_bilinear(i);

                    let next = rc.cache.get(next_probe);
                    *num_next += weight * next.loaded().select(1.0_f32.expr(), 0.0_f32.expr());
                }
                inferno(**num_next)*/
            } else if mode == 3 {
                /*let node = rc.cache.get(probe);
                if node.loaded() {
                    let encoder = DirectionIndexer::from_comps_expr(DirectionIndexerComps {
                        resolution: (rc.cache.base_angular_resolution / 2) << cascade,
                    });
                    let total_radiance = Vec4::splat(0.0_f32).var();
                    for j in 0_u32.expr()..encoder.linear_size() {
                        let dir = encoder.linear_to_dir(j);
                        let weight = dir.dot(normal);
                        if weight > 0.0 {
                            let radiance = rc.radiance.read(node.data_offset / 4 + j);
                            if radiance.is_finite().all() {
                                *total_radiance += weight * radiance.extend(1.0);
                            }
                        }
                    }
                    let result = total_radiance.xyz() / total_radiance.w;
                    result + emission
                } else {
                    Vec3::expr(1.0, 0.0, 0.0)
                }*/
                pcg3df(probe.position.xyz())
            } else if mode == 4 {
                let ray_start_pos = offset_ray(world_pos, normal);

                let added_radiance = Vec4::splat(0.0_f32).var();
                let num_samples = 128;
                for i in 0u32..num_samples {
                    let bounce_ray = sample_bounce(
                        normal,
                        (r2(tick * num_samples + i) + pcg3df(pixel.extend(968)).xy()).fract(),
                    );
                    // let bounce_ray = project_line(
                    //     (r2(tick * num_samples + i) + pcg3df(pixel.extend(968)).xy()).fract(),
                    // );
                    // let bounce_ray = bounce_ray * bounce_ray.dot(normal).signum();

                    let bounce_hit = core.world.trace(ray_start_pos, bounce_ray);
                    let radiance = core.get_radiance(None, ray_start_pos, bounce_ray, bounce_hit);
                    *added_radiance += radiance.extend(1.0); // * bounce_ray.dot(normal);
                }
                let value = accum_buf.read(pixel) + added_radiance;
                accum_buf.write(pixel, value);
                value.xyz() / value.w * diffuse + emission
            } else if mode == 5 {
                let node = rc.cache.get(probe);
                if node.loaded() {
                    let count = trace.ray_count.read(node.index).cast_f32();
                    let target = TARGET_QUALITY_MULTIPLIER
                        * rc.cache.get_encoder(cascade).linear_size().cast_f32();
                    inferno(count / target)
                } else {
                    Vec3::splat_expr(0.0)
                }
                // inferno(1.0 - reinhard((rc.cache.dequantize(probe) - world_pos).length()))
            } else if mode == 6 {
                let hit = core.hit.read(pixel_index);
                if (cursor_pixel == pixel).all() {
                    let inst = hit.inst;
                    device_println!("Hit at {}, position {:?}", host!(inst), host!(world_pos));
                }
                pcg3df(Vec3::expr(hit.inst, 37, 926))
            } else if mode == 7 {
                /*let cost = rc.cache.cost(probe);
                if cost != u32::MAX {
                    if cost > 8 {
                        Vec3::expr(1.0, 1.0, 1.0)
                    } else {
                        plasma(cost.cast_f32() / 9.0)
                    }
                } else {
                    Vec3::splat_expr(0.0)
                }*/
                Vec3::splat_expr(
                    1.0 - splat
                        .cm1_shadow_mask
                        .read(pixel_index)
                        .count_ones()
                        .cast_f32()
                        / rc.cache.directions_per_probe(0) as f32,
                )
            } else if mode == 8 {
                /*let z = pixel.x / ddgi.irradiance.width()
                    + (pixel.y / ddgi.irradiance.height())
                        * (app.width() / ddgi.irradiance.width());
                inferno(reinhard(
                    ddgi.distance
                        .read(
                            (pixel % Vec2::expr(ddgi.irradiance.width(), ddgi.irradiance.height()))
                                .extend(z),
                        )
                        .x,
                ))*/
                ddgi.query_irradiance(world_pos, normal, -ray_dir) * diffuse + emission
            } else {
                Vec3::expr(1.0, 0.0, 0.0)
            };
            app.set_pixel(pixel.cast_i32(), color);
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: scene.camera_yaw,
        pitch: scene.camera_pitch,
        pos: scene.camera_pos,
        fov: 1.15,
    };
    let mut running = true;
    let mut mode = 0;
    let mut draw_cascade = 0;
    let mut c0_len_mode = 0;
    let mut splat_mode = 0;
    let mut blend_radius = 0.5;
    let mut normal_rc = false;
    let mut secondary_bounce = false;
    let mut specular_mode = 0;

    let mut profiler = Profiler::new();

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
            mode = (mode + 1) % 9;
            println!("Drawing mode {mode}");
            let tonemap = matches!(mode, 0 | 1 | 2 | 3 | 4 | 8);
            rt.tonemap(tonemap);
        }
        if rt.key_pressed(KeyCode::KeyL) {
            draw_cascade = (draw_cascade + 1) % rc.num_cascades;
            println!("Level set {draw_cascade}");
        }
        if rt.key_pressed(KeyCode::KeyK) {
            c0_len_mode = (c0_len_mode + 1) % 5;
        }
        if rt.key_pressed(KeyCode::KeyO) {
            running = !running;
            println!("Running: {running}");
        }
        if rt.key_pressed(KeyCode::KeyB) {
            splat_mode = (splat_mode + 1) % 3;
        }
        if rt.key_pressed(KeyCode::BracketRight) {
            blend_radius += 0.01;
            if blend_radius > 1.0 {
                blend_radius = 0.5;
            }
        }
        if rt.key_pressed(KeyCode::KeyN) {
            normal_rc = !normal_rc;
        }
        if rt.key_pressed(KeyCode::KeyV) {
            secondary_bounce = !secondary_bounce;
        }
        if rt.key_pressed(KeyCode::KeyG) {
            specular_mode = (specular_mode + 1) % 3;
        }

        // spacing between end of rays shoudl be equal to spatial spacing.
        let inv_arc_size = rc.base_angular_resolution as f32 / PI;
        let inv_arc_size_accurate =
            ((2 * rc.base_angular_resolution * rc.base_angular_resolution) as f32 / (4.0 * PI))
                .sqrt();
        let c0_len = match c0_len_mode {
            0 => inv_arc_size_accurate,
            1 => inv_arc_size,
            2 => 4.0,
            3 => 9999.0,
            4 => 0.1,
            _ => unreachable!(),
        };

        let mut commands = CommandList::default();

        if rt.tick == 0 {
            rc.reset_table(&mut commands, false);
        }

        if rt.key_down(KeyCode::Backslash) {
            println!("Resetting cache");

            rc.reset_table(&mut commands, true);
        }

        commands += core.view.write_host(camera.view());

        commands += core.compute_gbuffer();

        let running = running || rt.key_pressed(KeyCode::Period);
        if running {
            if mode == 5 {
                println!("Num rays traced: {}", trace.next_traced_ray.read_blocking());
            }
            commands += rc.cache.set_view(camera.view(), rc.lod_bias);

            // add c0 probes
            commands += rc.init_probes_base(&core);
            if secondary_bounce {
                commands += rc.init_probes_secondary(&trace);
            }
            for c in 1..num_cascades {
                commands += rc.init_probes(c);
            }
            commands += rc.set_lifetime_base(&core);
            if secondary_bounce {
                commands += rc.set_lifetime_secondary(&trace);
            }
            for c in 1..num_cascades {
                commands += rc.set_lifetime(c);
            }

            rc.reset_table(&mut commands, false);
            rc.update_lod_bias();

            // the average probe positions are only read by normal RC and the splat depth prepass
            if normal_rc || splat_mode == 2 {
                commands += rc.compute_average_position(&core);
            }

            if normal_rc {
                for c in 0..num_cascades {
                    commands += rc.trace_deposit(&core, c0_len, c, 1, true);
                }
            } else {
                commands += trace.reset();

                commands += trace.compute_ray_count_base(&core, &rc);

                for c in 1..num_cascades {
                    commands += trace.compute_ray_count(&rc, c);
                }
                for c in (0..num_cascades).rev() {
                    commands += trace.compute_ray_offset(&rc, c);
                }
                if secondary_bounce {
                    commands += trace.compute_secondary_rays(&core, &rc, rt.tick, c0_len, true);
                }
                commands += trace.compute_rays(&core, &rc, rt.tick, c0_len, secondary_bounce);

                commands.execute();
                let ray_count = trace.next_traced_ray.read_blocking();

                commands += rc.deposit_split(trace.traced_rays.view(..ray_count as usize));
            }

            for c in (0..num_cascades).rev() {
                commands += rc.merge_cache(c);
            }
        }

        if camera_updated {
            commands.push(
                dispatch0(
                    track!(|| {
                        let pixel = dispatch_id().xy();
                        accum_buf.write(pixel, Vec4::splat_expr(0.0_f32));
                    }),
                    [accum_buf.width(), accum_buf.height(), 1],
                )
                .debug("Reset Accumulation"),
            );
            // commands += ris.reset();
        }

        commands += splat.reset();

        if rc.cm1_len > 0.0 {
            commands += splat.trace_cm1(&core, &rc, 4);
        }

        if mode == 0 {
            if rc.base_angular_resolution == 2 && splat_mode == 0 && draw_cascade == 0 {
                commands += splat.shade_optimized_8(&core, &rc);
            } else if rc.base_angular_resolution == 4
                && splat_mode == 0
                && draw_cascade == 0
                && rc.cm1_len == 0.0
            {
                commands += splat.calculate_irradiance(&rc);
                commands += splat.copy_irradiance_borders(&rc);
                commands += splat.shade_optimized_32(&core, &rc);
            } else {
                let blend_radius = match splat_mode {
                    0 => 1.0,
                    1 => blend_radius,
                    2 => 0.4,
                    3 => 0.51,
                    _ => unreachable!(),
                };
                let depth_prepass = splat_mode == 2;
                let dispatch_size = 32 << (draw_cascade.saturating_sub(1) * 2);
                if depth_prepass {
                    commands.push(
                        splat
                            .splat_radiance_raw(&core, &rc)
                            .dispatch_async(
                                [rc.max_probes[draw_cascade as usize], dispatch_size, 1],
                                &blend_radius,
                                &draw_cascade,
                                &match splat_mode {
                                    2 => 2,
                                    _ => unreachable!(),
                                },
                            )
                            .debug("Draw Splats (Depth Prepass)"),
                    );
                }
                commands.push(
                    splat
                        .splat_radiance_raw(&core, &rc)
                        .dispatch_async(
                            [rc.max_probes[draw_cascade as usize], dispatch_size, 1],
                            &blend_radius,
                            &draw_cascade,
                            &match splat_mode {
                                0 => 0,
                                1 => 0,
                                2 => 1,
                                _ => unreachable!(),
                            },
                        )
                        .debug("Draw Splats"),
                );
            }
        }
        if mode == 8 && !rt.key_down(KeyCode::KeyR) {
            commands += ddgi.trace_rays(&core, rt.tick);
            commands += ddgi.splat_irradiance();
            commands += ddgi.splat_distance();
            commands += ddgi.copy_borders();
            if rt.key_pressed(KeyCode::KeyT) {
                commands += ddgi.adjust_probe_offsets();
            }
            // commands += ddgi.draw_probes(&core, rt.overlay());
        }

        commands.push(
            draw_kernel
                .dispatch_async(
                    rt.dispatch_size(),
                    &camera.view(),
                    &rt.tick,
                    &(if splat_mode == 2 { DEBUG_SPLATS } else { 0 }
                        | match specular_mode {
                            0 => DRAW_DIFFUSE,
                            1 => DRAW_SPECULAR,
                            2 => DRAW_DIFFUSE | DRAW_SPECULAR,
                            _ => unreachable!(),
                        }),
                    &mode,
                    &draw_cascade,
                    &rt.cursor_position.map(|x| x as u32),
                )
                .debug("Draw"),
        );

        commands.execute();
        profiler.record(commands.timings);

        if profiler.time() > 1000.0 {
            profiler.print(
                &[
                    "Compute G-Buffer",
                    "Init Probes",
                    "Build Freelists",
                    "Compute Ray Count",
                    "Compute Ray Offset",
                    "Compute Rays",
                    "Deposit Split",
                    "Trace and Deposit",
                    "DDGI",
                    "Merge",
                    "Trace Cascade -1",
                    "Calculate Irradiance",
                    "Draw",
                    "Secondary",
                ],
                true,
            );
            profiler.reset();
            println!("Num probes: {:?}", rc.num_probes);
            println!("Max probe: {:?}", rc.max_probes);
            println!(
                "Limit: {:?}",
                (0..num_cascades)
                    .map(|n| rc.cache.max_probes_at_cascade(n))
                    .collect::<Vec<_>>()
            );
            let bytes_used = rc.cache.table.memory_size()
                + rc.probe_lifetime.size_bytes()
                + rc.probe_quality.size_bytes()
                // + rc.base_radiance.size_bytes()
                // + rc.radiance.size_bytes()
                // + rc.cache.weighted_fluence.size_bytes()
                // + rc.cache.weights.size_bytes()
            + (std::mem::size_of::<Vec4<f32>>() + std::mem::size_of::<f32>()
            + std::mem::size_of::<Vec4<f32>>() / 4)
                * rc.num_probes
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * rc.cache.directions_per_probe(i as u32))
                    .sum::<u32>() as usize
            + std::mem::size_of::<Vec4<f32>>()
                * rc.num_probes[0] as usize
                * rc.cache.directions_per_probe(0) as usize;
            println!("Memory: {:?} MB", bytes_used / 1024 / 1024);
            println!(
                "Table size: {:?} MB",
                rc.cache.table.memory_size() / 1024 / 1024
            );
            if mode == 8 {
                println!("Num used: {:?}", ddgi.calc_probes_used());
            }
        }

        #[cfg(feature = "denoise")]
        if rt.key_pressed(KeyCode::KeyI) {
            println!("Denoising!");
            denoiser.denoise(&core, &trace.traced_hits);
        }

        flush_printer();
    });
}
