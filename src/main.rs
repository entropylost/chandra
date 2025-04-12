use glam::{Mat3 as FMat3, UVec3};
use hrc::storage::{BufferStorage, RadianceStorage};
use hrc::trace::{StorageTracer, Tracer, VoxelTracer, WorldMapper, merge_up};
use hrc::{Axis, Grid, Probe};
use keter::lang::types::vector::{Mat3, Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};

mod hrc;
mod utils;
use noise::{Fbm, NoiseFn, Perlin};
use utils::{dda, to_linear, to_linear_expr};

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
struct View {
    screen_size: Vec2<f32>,
    pos: Vec3<f32>,
    transform: Mat3,
}

#[derive(Debug, Clone, Copy)]
struct Camera {
    screen_size: Vec2<f32>,
    yaw: f32,
    pitch: f32,
    pos: Vec3<f32>,
    fov: f32,
}
impl Camera {
    fn rotation(&self) -> FMat3 {
        let view = FMat3::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0).transpose();
        FMat3::from_cols(view.x_axis, view.z_axis, view.y_axis).transpose()
    }
    fn view(&self) -> View {
        let view = self.rotation();
        let ratio = (self.fov / 2.0).tan() / (self.screen_size.y / 2.0);
        let view = FMat3::from_cols(view.x_axis * ratio, -view.y_axis * ratio, view.z_axis);

        View {
            screen_size: self.screen_size,
            pos: self.pos,
            transform: view.into(),
        }
    }
    fn forward(&self) -> Vec3<f32> {
        self.rotation().z_axis.into()
    }
    fn right(&self) -> Vec3<f32> {
        self.rotation().x_axis.into()
    }
}

type Color = hrc::color::Color<hrc::color::RgbF16Single>;
type Fluence = hrc::fluence::Fluence<hrc::fluence::RgbF16Single>;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Draw {
    Circle(f32),
    Rect(f32, f32, f32),
}
impl Draw {
    fn size(self) -> Vec3<f32> {
        match self {
            Self::Circle(r) => Vec3::splat(r),
            Self::Rect(x, y, z) => Vec3::new(x, y, z),
        }
    }
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [2048; 2])
        .agx()
        .dither(8)
        .resize()
        .init();

    let palette = DEVICE.create_buffer_from_slice(&[Vec3::new(1.0, 1.0, 1.0); 12]);
    let k = 0.05;
    let color_palette = DEVICE.create_buffer_from_slice(&[
        Color::dark(0.002),
        Color::solid(Vec3::splat(f16::ZERO)),
        Color::solid(Vec3::splat(f16::from_f32(5.0))),
        Color::dark(0.0 * k),
        Color::dark(1.0 * k),
        Color::dark(2.0 * k),
        Color::dark(3.0 * k),
        Color::dark(4.0 * k),
        Color::dark(5.0 * k),
        Color::dark(6.0 * k),
        Color::dark(7.0 * k),
        Color::dark(8.0 * k),
    ]);

    let base_size = 64_u32;
    let size = UVec3::splat(base_size);

    let mut voxels = vec![0; size.element_product() as usize];

    let fbm = Fbm::<Perlin>::new(150392);

    for pos in (0..base_size).flat_map(|x| {
        (0..base_size).flat_map(move |y| (0..base_size).map(move |z| UVec3::new(x, y, z)))
    }) {
        let value = fbm.get([
            pos.x as f64 / 32.0,
            pos.y as f64 / 32.0,
            pos.z as f64 / 32.0,
        ]) as f32;
        let value = (value + 1.0) / 2.0 - pos.z as f32 / base_size as f32 * 1.4;
        let value = value.clamp(0.0, 0.999);
        let index = (value * 9.0).floor() as u8;
        voxels[to_linear(pos, size)] = index + 3;
    }

    for draw in [
        ([0.0, 32.0, 32.0], Draw::Rect(1.0, 32.0, 32.0), 1),
        ([64.0, 32.0, 32.0], Draw::Rect(1.0, 32.0, 32.0), 1),
        ([32.0, 64.0, 32.0], Draw::Rect(32.0, 1.0, 32.0), 1),
        ([32.0, 32.0, 0.0], Draw::Rect(32.0, 32.0, 1.0), 1),
        ([32.0, 32.0, 64.0], Draw::Rect(32.0, 32.0, 1.0), 1),
        ([64.0, 32.0, 32.0], Draw::Rect(3.0, 8.0, 8.0), 2),
        ([32.0, 32.0, 16.0], Draw::Circle(16.0), 1),
    ] {
        let (center, draw, color): ([f32; 3], Draw, u8) = draw;
        let center = Vec3::from(center);
        let min = (center - draw.size()).map(|x| (x.floor().max(0.0) as u32).min(base_size - 1));
        let max = (center + draw.size()).map(|x| (x.ceil().max(0.0) as u32).min(base_size - 1));
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    let pos = Vec3::new(x, y, z);
                    let offset = pos.map(|x| x as f32 + 0.5) - center;
                    if match draw {
                        Draw::Circle(r) => {
                            offset.map(|x| x * x).elements.into_iter().sum::<f32>() < r * r
                        }
                        Draw::Rect(x, y, z) => {
                            offset.x.abs() <= x && offset.y.abs() <= y && offset.z.abs() <= z
                        }
                    } {
                        voxels[to_linear(pos.into(), size)] = color;
                    }
                }
            }
        }
    }

    // for voxel in &model.voxels {
    //     let pos = UVec3::new(voxel.x as u32, voxel.y as u32, voxel.z as u32);
    //     if pos.cmpge(size).any() {
    //         continue;
    //     }
    //     voxels[to_linear(pos, size)] = voxel.i;
    // }
    let voxels = DEVICE.create_buffer_from_slice(&voxels);

    let size = Vec3::from(size);

    let base_grid = Grid::new(Vec3::splat(base_size), 2);

    let ray_pyramid = (0..base_size.trailing_zeros())
        .map(|level| {
            let mut grid = base_grid.shift(level);
            grid.directions += 1;
            DEVICE.create_buffer::<Fluence>(grid.size() as usize)
        })
        .collect::<Vec<_>>();

    let radiance_pyramid = (0..base_size.trailing_zeros())
        .map(|level| {
            let grid = base_grid.shift(level);
            DEVICE.create_buffer::<Vec3<f16>>(grid.size() as usize)
        })
        .collect::<Vec<_>>();

    let final_radiance =
        DEVICE.create_buffer::<Vec3<f16>>((base_size * base_size * base_size) as usize);

    let voxel_tracer = VoxelTracer {
        palette: color_palette.view(..),
        buffer: voxels.view(..),
        size,
    };
    let axes = [
        Axis::CellX,
        Axis::CellY,
        Axis::CellZ,
        Axis::DirectionU,
        Axis::DirectionV,
    ];

    let store = DEVICE.create_kernel::<fn(u32, Buffer<Fluence>, Mat3, Vec3<f32>)>(&track!(
        |level, buffer, rotation, offset| {
            let grid = base_grid.expr().shift(level);
            let tracer = WorldMapper::<hrc::fluence::RgbF16Single, _> {
                tracer: &voxel_tracer,
                grid,
                rotation,
                offset,
                _marker: Default::default(),
            };

            let (cell, dir) = Axis::extract(
                axes,
                (grid.size, Vec2::splat_expr(grid.directions + 1)),
                dispatch_id().x,
            );
            let probe = Probe::expr(cell.cast_i32(), dir);

            let fluence = tracer.compute_ray(probe);

            buffer.write(dispatch_id().x, fluence);
        }
    ));
    let merge_up = DEVICE.create_kernel::<fn(u32, Buffer<Fluence>, Buffer<Fluence>)>(&track!(
        |level, buffer, next_buffer| {
            let grid = base_grid.expr().shift(level);
            let next_grid = grid.next();

            let (cell, dir) = Axis::extract(
                axes,
                (next_grid.size, Vec2::splat_expr(next_grid.directions + 1)),
                dispatch_id().x,
            );
            let next_probe = Probe::expr(cell.cast_i32(), dir);
            let storage = StorageTracer {
                axes,
                grid,
                buffer,
                next_buffer: next_buffer.clone(),
            };
            let fluence = merge_up(&storage, next_probe);
            next_buffer.write(dispatch_id().x, fluence);
        }
    ));

    let merge = DEVICE.create_kernel::<fn(
        u32,
        Buffer<Fluence>,
        Buffer<Fluence>,
        Buffer<Vec3<f16>>,
        Buffer<Vec3<f16>>,
    )>(&track!(|level,
                                      ray_buffer,
                                      next_ray_buffer,
                                      buffer,
                                      next_buffer| {
        let grid = base_grid.expr().shift(level);

        let tracer = StorageTracer {
            axes,
            grid,
            buffer: ray_buffer,
            next_buffer: next_ray_buffer,
        };

        let (cell, dir) = Axis::extract(
            axes,
            (grid.size, Vec2::splat_expr(grid.directions)),
            dispatch_id().x,
        );
        let probe = Probe::expr(cell.cast_i32(), dir);
        let trace = tracer.trace(probe);

        let storage = BufferStorage {
            axes,
            grid: grid.next(),
            buffer: next_buffer,
        };
        let radiance = hrc::merge(grid, probe, trace, &storage);
        BufferStorage { axes, grid, buffer }.store(probe, radiance);
    }));
    let copy_final = DEVICE.create_kernel::<fn(Mat3, Vec3<f32>)>(&track!(|rotation, offset| {
        let storage = BufferStorage {
            axes,
            grid: base_grid.expr(),
            buffer: radiance_pyramid[0].var(),
        };
        let radiance = Vec3::splat(f16::ZERO).var();
        for quad in hrc::quads() {
            let probe = Probe::expr(
                dispatch_id().cast_i32() + Vec3::new(quad.x as i32, quad.y as i32, 0),
                Vec2::new(quad.x as u32, quad.y as u32).expr(),
            );
            *radiance += storage.load(probe);
        }
        // TODO: Add final-voxel tracing. Also make surface lookups work properly.
        let pos_norm = dispatch_id().cast_f32() / base_size as f32;
        let pos = rotation * pos_norm + offset;
        let pos = pos.cast_u32();
        let index = to_linear_expr(pos, size);
        final_radiance.write(index, final_radiance.read(index) + radiance);
    }));

    let dirs = [UVec3::X, UVec3::Y, UVec3::Z];
    let transforms = (0..3).flat_map(|i| {
        let dir = dirs[i];
        let perp1 = dirs[(i + 1) % 3];
        let perp2 = dirs[(i + 2) % 3];
        let rotation = Mat3::from(
            base_size as f32 * FMat3::from_cols(perp1.as_vec3(), perp2.as_vec3(), dir.as_vec3()),
        );
        let opposite_rotation = Mat3::from(
            base_size as f32 * FMat3::from_cols(perp2.as_vec3(), perp1.as_vec3(), -dir.as_vec3()),
        );
        [
            (
                rotation,
                Vec3::splat(0.5),
                Vec3::from(dir.as_vec3()) * 1.5 + Vec3::splat(0.001),
            ),
            (
                opposite_rotation,
                Vec3::splat(0.5) + Vec3::from(dir.as_vec3()) * (base_size as f32 - 1.0),
                Vec3::from(dir.as_vec3()) * (base_size as f32 - 1.5) + Vec3::splat(0.001),
            ),
        ]
    });

    println!(
        "Size: {:?}gb",
        (radiance_pyramid.iter().map(|x| x.len()).sum::<usize>() * std::mem::size_of::<Vec3<f16>>()
            + ray_pyramid.iter().map(|x| x.len()).sum::<usize>() * std::mem::size_of::<Fluence>())
            as f32
            / (1024.0_f32).powi(3)
    );

    let start = std::time::Instant::now();

    for (rotation, offset, trace_offset) in transforms {
        for level in 0..base_size.trailing_zeros() {
            let mut grid = base_grid.shift(level);
            grid.directions += 1;
            if level == 0 || level == 1 {
                store.dispatch(
                    [grid.size(), 1, 1],
                    &level,
                    &ray_pyramid[level as usize],
                    &rotation,
                    &trace_offset,
                );
            } else {
                merge_up.dispatch(
                    [grid.size(), 1, 1],
                    &(level - 1),
                    &ray_pyramid[(level - 1) as usize],
                    &ray_pyramid[level as usize],
                );
            }
        }
        for level in (0..base_size.trailing_zeros() - 1).rev() {
            merge.dispatch(
                [base_grid.shift(level).size(), 1, 1],
                &level,
                &ray_pyramid[level as usize],
                &ray_pyramid[(level + 1) as usize],
                &radiance_pyramid[level as usize],
                &radiance_pyramid[(level + 1) as usize],
            );
        }
        copy_final.dispatch([base_size, base_size, base_size], &rotation, &offset);
    }
    DEVICE.default_stream().scope().synchronize();

    println!("Time: {:?}", start.elapsed());

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32)>(
        KernelBuildOptions {
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view, slice| {
            set_block_size([8, 8, 1]);
            let pixel = dispatch_id().xy();
            let ray_dir = view.transform * (pixel.cast_f32() - view.screen_size / 2.0).extend(1.0);
            let ray_dir = ray_dir.normalize();

            let last_pos = view.pos.floor().cast_i32().var();

            let ambient = Fluence::empty().var();

            let hit = {
                let time = f32::INFINITY.var();
                dda(
                    view.pos,
                    ray_dir,
                    10000.0.expr(),
                    (Vec3::splat_expr(0.01), size.expr().cast_f32() - 0.01),
                    |pos, last_t, next_t| {
                        let v = voxels.read(to_linear_expr(pos, Vec3::from(size)));
                        let color = color_palette.read(v.cast_u32());
                        if (slice != 0 && pos.y == slice) || (slice == 0 && color.opacity > 5.0) {
                            *time = (last_t + next_t) / 2.0;
                            true.expr()
                        } else {
                            let segment_size = next_t - last_t;
                            let radiance = final_radiance.read(to_linear_expr(pos, size));
                            let diffuse: Expr<Vec3<f32>> = palette.read(v.cast_u32());
                            *ambient = ambient.over(
                                hrc::color::Color::<hrc::color::RgbF16Single>::expr(
                                    radiance * diffuse.cast_f16() + color.emission,
                                    color.opacity,
                                )
                                .to_fluence(segment_size),
                            );
                            *last_pos = pos.cast_i32();
                            false.expr()
                        }
                    },
                );
                **time
            };

            if hit != f32::INFINITY {
                let world_pos = view.pos + ray_dir * hit;
                let world_voxel = world_pos.floor().cast_i32();
                let prev_voxel = if slice == 0 { **last_pos } else { world_voxel };

                let radiance =
                    if (prev_voxel < 0).any() || (prev_voxel >= size.expr().cast_i32()).any() {
                        Vec3::splat_expr(f16::ZERO)
                    } else {
                        final_radiance.read(to_linear_expr(
                            prev_voxel.clamp(0, size.expr().cast_i32() - 1).cast_u32(),
                            size,
                        ))
                    } + f16::from_f32(0.01);

                let color = voxels.read(to_linear_expr(world_voxel.cast_u32(), Vec3::from(size)));
                // Vec3::splat_expr(num_ddas.cast_f32() * 0.1); // pcg3df(Vec3::expr(hit.inst, hit.prim, 1));
                let color = palette.read(color.cast_u32());
                app.set_pixel(
                    pixel.cast_i32(),
                    if (color != Vec3::splat(0.0)).any() {
                        ambient.radiance.cast_f32()
                            + ambient.transmittance * color * radiance.cast_f32()
                    } else {
                        radiance.cast_f32()
                    },
                );
            } else {
                app.set_pixel(pixel.cast_i32(), ambient.radiance.cast_f32());
            }
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 0.0,
        pitch: 0.0,
        pos: Vec3::new(32.0, -64.0, 32.0),
        fov: 0.9,
    };

    let mut slice: u32 = 0;

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            5.0
        } else {
            200.0
        } * rt.frame_time();
        for key in &rt.keys_down {
            match key {
                KeyCode::ArrowLeft => camera.yaw -= 1.0 * rot_speed,
                KeyCode::ArrowRight => camera.yaw += 1.0 * rot_speed,
                KeyCode::ArrowUp => camera.pitch -= 1.0 * rot_speed,
                KeyCode::ArrowDown => camera.pitch += 1.0 * rot_speed,
                KeyCode::KeyW => camera.pos = camera.pos + camera.forward() * move_speed,
                KeyCode::KeyS => camera.pos = camera.pos - camera.forward() * move_speed,
                KeyCode::KeyA => camera.pos = camera.pos - camera.right() * move_speed,
                KeyCode::KeyD => camera.pos = camera.pos + camera.right() * move_speed,
                KeyCode::Space => camera.pos = camera.pos + Vec3::new(0.0, 0.0, 1.0) * move_speed,
                KeyCode::ShiftLeft => {
                    camera.pos = camera.pos - Vec3::new(0.0, 0.0, 1.0) * move_speed
                }
                _ => (),
            }
        }
        if rt.key_pressed(KeyCode::Digit1) {
            slice = 0;
        }
        if rt.key_pressed(KeyCode::KeyE) {
            slice += 1;
            println!("Slice: {:?}", slice);
        }
        if rt.key_pressed(KeyCode::KeyQ) {
            slice = slice.saturating_sub(1);
            println!("Slice: {:?}", slice);
        }

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view(), &slice);
        // draw_lines.dispatch([lines.len() as u32, 256, 1], &camera.view());
    });
}
