use glam::{Mat3 as FMat3, UVec3};
use hrc::storage::{BufferStorage, RadianceStorage};
use hrc::trace::{VoxelTracer, WorldMapper};
use hrc::{Axis, Grid, Probe};
use keter::lang::types::vector::{Mat3, Vec2, Vec3};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};

mod hrc;
mod utils;
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

type Color = hrc::color::Color<hrc::color::BinaryF32>;
type Fluence = hrc::fluence::Fluence<hrc::fluence::BinaryF32>;

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [2048; 2]).agx().resize().init();

    let file = dot_vox::load("assets/test.vox").unwrap();

    let palette = DEVICE.create_buffer_from_fn(file.palette.len(), |i| {
        let color = file.palette[i];
        Vec3::new(color.r as f32, color.g as f32, color.b as f32).map(|x| (x / 255.0).powf(2.2))
    });
    let color_palette = DEVICE.create_buffer_from_fn(file.palette.len(), |i| {
        if i == 0 {
            Color::solid(10.0_f32)
        } else {
            Color::solid(0.0_f32)
        }
    });

    let model = &file.models[0];
    let size = UVec3::new(64, 64, 64);

    let mut voxels = vec![255; size.element_product() as usize];
    for voxel in &model.voxels {
        let pos = UVec3::new(voxel.x as u32, voxel.y as u32, voxel.z as u32);
        // if pos.cmpeq(UVec3::splat(0)).any() || pos.cmpge(size - 1).any() {
        //     continue;
        // }
        voxels[to_linear(pos, size)] = voxel.i;
    }
    let voxels = DEVICE.create_buffer_from_slice(&voxels);

    let size = Vec3::from(size);

    let base_size = 64_u32;
    let base_grid = Grid::new(Vec3::splat(base_size), 1);
    // Don't actually need the =, again.
    let buffer_pyramid = (0..=base_size.trailing_zeros())
        .map(|level| {
            let grid = base_grid.shift(level);
            DEVICE.create_buffer::<f32>(grid.size() as usize)
        })
        .collect::<Vec<_>>();

    let final_radiance = DEVICE.create_buffer::<f32>((base_size * base_size * base_size) as usize);

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

    let merge = DEVICE.create_kernel::<fn(u32, Buffer<f32>, Buffer<f32>, Mat3, Vec3<f32>)>(
        &track!(|level, buffer, next_buffer, rotation, offset| {
            let tracer = WorldMapper::<hrc::fluence::BinaryF32, _> {
                tracer: &voxel_tracer,
                rotation,
                offset,
                _marker: Default::default(),
            };

            let grid = Grid::expr(
                Vec3::expr(base_size, base_size, base_size >> level),
                1 << level,
            );
            let probe = Axis::extract(
                axes,
                (grid.size, Vec2::splat_expr(grid.directions)),
                dispatch_id().x,
            );
            let probe = Probe::expr(probe.0.cast_i32(), probe.1);
            let storage = BufferStorage {
                axes,
                buffer: next_buffer,
            };
            let radiance = hrc::merge(grid, probe, &storage, &tracer);
            BufferStorage { axes, buffer }.store(grid, probe, radiance);
        }),
    );
    let copy_final = DEVICE.create_kernel::<fn(Mat3, Vec3<f32>)>(&track!(|rotation, offset| {
        let probe = Probe::expr(dispatch_id().cast_i32(), Vec2::splat_expr(0));
        let radiance = BufferStorage {
            axes,
            buffer: buffer_pyramid[0].var(),
        }
        .load(base_grid.expr(), probe);
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
            (rotation, Vec3::splat(0.5)),
            (
                opposite_rotation,
                Vec3::splat(0.5) + Vec3::from(dir.as_vec3()) * base_size as f32,
            ),
        ]
    });

    for (rotation, offset) in transforms {
        for level in (0..base_size.trailing_zeros()).rev() {
            merge.dispatch(
                [base_grid.shift(level).size(), 1, 1],
                &level,
                &buffer_pyramid[level as usize],
                &buffer_pyramid[(level + 1) as usize],
                &rotation,
                &offset,
            );
        }
        copy_final.dispatch([base_size, base_size, base_size], &rotation, &offset);
    }

    // let test_setup = DEVICE.create_kernel::<fn()>(&track!(|| {
    //     let grid = base_grid.expr();
    //     let probe = Axis::extract(
    //         axes,
    //         (grid.size, Vec2::splat_expr(grid.directions)),
    //         dispatch_id().x,
    //     );
    //     let probe = Probe::expr(probe.0.cast_i32(), probe.1);
    //
    //     BufferStorage {
    //         axes,
    //         buffer: buffer_pyramid[0].var(),
    //     }
    //     .store(grid, probe, 1.0.expr());
    // }));
    // test_setup.dispatch([base_size * base_size * base_size, 1, 1]);

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View, u32)>(
        KernelBuildOptions {
            name: Some("draw_kernel_4".into()),
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

            let hit = {
                let time = f32::INFINITY.var();
                dda(
                    view.pos,
                    ray_dir,
                    10000.0.expr(),
                    (Vec3::splat_expr(0.01), size.expr().cast_f32() - 0.01),
                    |pos, last_t, next_t| {
                        if slice != 0 {
                            if pos.y == slice {
                                *time = (last_t + next_t) / 2.0;

                                true.expr()
                            } else {
                                false.expr()
                            }
                        } else {
                            if voxels.read(to_linear_expr(pos, Vec3::from(size))) != 255 {
                                *time = (last_t + next_t) / 2.0;
                                true.expr()
                            } else {
                                *last_pos = pos.cast_i32();
                                false.expr()
                            }
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
                        0.0_f32.expr()
                    } else {
                        final_radiance.read(to_linear_expr(
                            prev_voxel.clamp(0, size.expr().cast_i32() - 1).cast_u32(),
                            size,
                        ))
                    } + 0.01;

                let color = voxels.read(to_linear_expr(world_voxel.cast_u32(), Vec3::from(size)));
                // Vec3::splat_expr(num_ddas.cast_f32() * 0.1); // pcg3df(Vec3::expr(hit.inst, hit.prim, 1));
                let color = palette.read(color.cast_u32());
                app.set_pixel(
                    pixel.cast_i32(),
                    if (color != Vec3::splat(0.0)).any() {
                        color * radiance
                    } else {
                        Vec3::splat_expr(radiance)
                    },
                );
            }
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 0.8,
        pitch: 0.4,
        pos: Vec3::new(-30.0, -30.0, 60.0),
        fov: 1.15,
    };

    let mut slice: u32 = 0;

    app.run(|rt| {
        let rot_speed = 1.5 * rt.frame_time();
        let move_speed = if rt.key_down(KeyCode::ControlLeft) {
            5.0
        } else {
            200.0
        } * rt.frame_time();
        rt.process_keys_down(|key| match key {
            KeyCode::ArrowLeft => camera.yaw -= 1.0 * rot_speed,
            KeyCode::ArrowRight => camera.yaw += 1.0 * rot_speed,
            KeyCode::ArrowUp => camera.pitch -= 1.0 * rot_speed,
            KeyCode::ArrowDown => camera.pitch += 1.0 * rot_speed,
            KeyCode::KeyW => camera.pos = camera.pos + camera.forward() * move_speed,
            KeyCode::KeyS => camera.pos = camera.pos - camera.forward() * move_speed,
            KeyCode::KeyA => camera.pos = camera.pos - camera.right() * move_speed,
            KeyCode::KeyD => camera.pos = camera.pos + camera.right() * move_speed,
            KeyCode::Space => camera.pos = camera.pos + Vec3::new(0.0, 0.0, 1.0) * move_speed,
            KeyCode::ShiftLeft => camera.pos = camera.pos - Vec3::new(0.0, 0.0, 1.0) * move_speed,
            _ => (),
        });
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

        rt.log_fps();

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view(), &slice);
        // draw_lines.dispatch([lines.len() as u32, 256, 1], &camera.view());
    });
}
