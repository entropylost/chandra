use std::mem::forget;

use dot_vox::{DotVoxData, SceneNode};
use glam::{IVec3, Mat3 as FMat3, UVec3, Vec3 as FVec3};
use keter::lang::types::vector::{Mat3, Mat4, Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter::rtx::{Aabb, AccelBuildRequest, AccelOption, AccelTraceOptions, AccelUsageHint, Ray};
use keter_testbed::{App, KeyCode};

mod hrc;
mod utils;
use utils::{dda, iter_vec, to_linear};

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

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
struct RenderedModel {
    position: Vec3<f32>,
    rotation: Mat3,
    index: u32,
}

fn to_transform(position: IVec3, rotation: dot_vox::Rotation, size: UVec3) -> (Vec3<f32>, Mat3) {
    let position = position.as_vec3();

    let rotation = FMat3::from_cols_array_2d(&rotation.to_cols_array_2d());

    let mut offset = FVec3::select(
        (size % 2).cmpeq(UVec3::ZERO),
        FVec3::ZERO,
        FVec3::splat(0.5),
    );
    offset = rotation.mul_vec3(offset); // If another seam shows up in the future, try multiplying this with `scale`
    let center = rotation * (size.as_vec3() / 2.0);
    ((position - center + offset).into(), rotation.into())
}

fn visit(
    file: &DotVoxData,
    rendered: &mut Vec<RenderedModel>,
    position: glam::IVec3,
    rotation: dot_vox::Rotation,
    index: u32,
) {
    if file.scenes.is_empty() && index == 0 {
        rendered.push(RenderedModel {
            index: 0,
            position: Vec3::splat(0.0),
            rotation: Mat3::identity(),
        });
        return;
    }
    match &file.scenes[index as usize] {
        SceneNode::Group { children, .. } => {
            for child in children {
                visit(file, rendered, position, rotation, *child);
            }
        }
        SceneNode::Shape { models, .. } => {
            for model in models {
                let size = file.models[model.model_id as usize].size;
                let (position, rotation) =
                    to_transform(position, rotation, glam::UVec3::new(size.x, size.y, size.z));
                rendered.push(RenderedModel {
                    index: model.model_id,
                    position,
                    rotation,
                });
            }
        }
        SceneNode::Transform { frames, child, .. } => {
            let frame = frames.first().unwrap();
            let position = frame
                .position()
                .map(|pos| glam::IVec3::new(pos.x, pos.y, pos.z))
                .unwrap_or(glam::IVec3::ZERO)
                + position;

            let rotation = frame.orientation().unwrap_or(dot_vox::Rotation::IDENTITY); // * rotation;

            visit(file, rendered, position, rotation, *child);
        }
    }
}

const BRICK_SIZE: u32 = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
struct BrickMask {
    values: u64,
}
impl BrickMask {
    fn empty() -> Self {
        Self { values: 0 }
    }
    fn set(&mut self, pos: UVec3, value: bool) {
        self.values |=
            (value as u64) << (pos.x + pos.y * BRICK_SIZE + pos.z * BRICK_SIZE * BRICK_SIZE);
    }
}
impl BrickMaskExpr {
    #[tracked]
    fn read(&self, pos: Expr<Vec3<u32>>) -> Expr<bool> {
        (self.values >> (pos.x + pos.y * BRICK_SIZE + pos.z * BRICK_SIZE * BRICK_SIZE).cast_u64())
            & 1
            != 0
    }
}

// do sudo nvidia-smi --lock-gpu-clocks=405,1500 before running to make it have fixed time.
// nvidia-smi -q -d SUPPORTED_CLOCKS
fn main() {
    let app = App::new("Sunfall", [2048; 2]).agx().resize().init();

    let file = dot_vox::load("assets/castle.vox").unwrap();

    let palette = DEVICE.create_buffer_from_fn(file.palette.len(), |i| {
        let color = file.palette[i];
        Vec3::new(color.r as f32, color.g as f32, color.b as f32).map(|x| (x / 255.0).powf(2.2))
    });

    let bindless = DEVICE.create_bindless_array(file.models.len());

    let mut brick_masks = Vec::<BrickMask>::new();
    let mut brick_colors = Vec::<u8>::new();
    let mut brick_positions = Vec::<Vec3<u32>>::new();
    let mut model_bvhs = Vec::new();
    let mut model_sizes = Vec::<Vec3<u32>>::new();
    let mut model_brick_offsets = Vec::<u32>::new();
    for (index, model) in file.models.iter().enumerate() {
        let s = model.size;
        let s = UVec3::new(s.x, s.y, s.z);
        let bricks = s.map(|x| x.div_ceil(BRICK_SIZE));
        let s = bricks * BRICK_SIZE;
        model_brick_offsets.push(brick_masks.len() as u32);
        let mut aabbs = Vec::<Aabb>::new();
        let mut brick_map = vec![u32::MAX; bricks.element_product() as usize];
        let mut voxels = vec![255; s.element_product() as usize];
        for voxel in &model.voxels {
            let pos = UVec3::new(voxel.x as u32, voxel.y as u32, voxel.z as u32);
            voxels[to_linear(pos, s)] = voxel.i;
        }
        if model.voxels.is_empty() {
            voxels[0] = 0;
        }
        for brick in iter_vec(bricks) {
            let mut mask = BrickMask::empty();
            let mut colors = [255; (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE) as usize];
            let mut min = UVec3::splat(BRICK_SIZE);
            let mut max = UVec3::splat(0);
            for brick_pos in iter_vec(UVec3::splat(BRICK_SIZE)) {
                let pos = brick * BRICK_SIZE + brick_pos;
                let voxel = voxels[to_linear(pos, s)];
                if voxel != 255 {
                    mask.set(brick_pos, true);
                    min = min.min(brick_pos);
                    max = max.max(brick_pos + 1);
                }
                colors[to_linear(brick_pos, UVec3::splat(BRICK_SIZE))] = voxel;
            }
            if mask != BrickMask::empty() {
                brick_map[to_linear(brick, bricks)] = brick_masks.len() as u32;
                brick_masks.push(mask);
                brick_colors.extend_from_slice(&colors);
                brick_positions.push((brick * BRICK_SIZE).into());
                aabbs.push(Aabb {
                    min: (brick * BRICK_SIZE + min).as_vec3().into(),
                    max: (brick * BRICK_SIZE + max).as_vec3().into(),
                });
            }
        }
        let brick_map = DEVICE.create_buffer_from_slice(&brick_map);
        bindless.emplace_buffer(index, &brick_map);
        forget(brick_map);
        if aabbs.is_empty() {
            println!("Voxels: {:?}", model.voxels);
            println!("Bricks: {:?}", bricks);
            println!("Size: {:?}", s);
            panic!("Shit");
        }
        let aabbs = DEVICE.create_buffer_from_slice(&aabbs);
        let bvh = DEVICE.create_procedural_primitive(
            aabbs.view(..),
            AccelOption {
                hint: AccelUsageHint::FastTrace,
                allow_compaction: true,
                allow_update: false,
            },
        );
        bvh.build(AccelBuildRequest::ForceBuild);
        model_bvhs.push(bvh);
        forget(aabbs);
        model_sizes.push(bricks.into());
        if index % 20 == 0 {
            println!("Loading: {:?}", index);
        }
    }
    println!("Loaded");
    let model_brick_offsets = DEVICE.create_buffer_from_slice(&model_brick_offsets);
    let brick_masks = DEVICE.create_buffer_from_slice(&brick_masks);
    let brick_colors = DEVICE.create_buffer_from_slice(&brick_colors);
    let brick_positions = DEVICE.create_buffer_from_slice(&brick_positions);
    // let model_sizes = DEVICE.create_buffer_from_slice(&model_sizes);

    let mut rendered_models = Vec::new();
    visit(
        &file,
        &mut rendered_models,
        glam::IVec3::ZERO,
        dot_vox::Rotation::IDENTITY,
        0,
    );

    let accel = DEVICE.create_accel(AccelOption {
        hint: AccelUsageHint::FastTrace,
        allow_compaction: true,
        allow_update: false,
    });

    for model in &rendered_models {
        fn extend(a: Vec3<f32>, b: f32) -> Vec4<f32> {
            Vec4::new(a.x, a.y, a.z, b)
        }
        accel.push_procedural_primitive(
            &model_bvhs[model.index as usize],
            Mat4 {
                cols: [
                    extend(model.rotation.cols[0], 0.0),
                    extend(model.rotation.cols[1], 0.0),
                    extend(model.rotation.cols[2], 0.0),
                    extend(model.position, 1.0),
                ],
            },
            0xFF,
        );
    }

    let rendered_models = DEVICE.create_buffer_from_slice(&rendered_models);
    accel.build(AccelBuildRequest::ForceBuild);

    let trace = track!(|ray_start: Expr<Vec3<f32>>, ray_dir: Expr<Vec3<f32>>| {
        accel
            .traverse(
                Ray::new_expr(
                    Expr::<[f32; 3]>::from(ray_start),
                    0.0,
                    Expr::<[f32; 3]>::from(ray_dir),
                    10000.0,
                ),
                AccelTraceOptions {
                    mask: 0xFF.expr(),
                    ..Default::default()
                },
            )
            .on_procedural_hit(|candidate| {
                let model = rendered_models.read(candidate.inst);
                let brick = model_brick_offsets.read(model.index) + candidate.prim;
                let mask = brick_masks.read(brick);

                let ray_start = model.rotation.transpose() * (ray_start - model.position)
                    - brick_positions.read(brick).cast_f32();
                let ray_dir = model.rotation.transpose() * ray_dir;

                dda(
                    ray_start,
                    ray_dir,
                    10000.0.expr(),
                    (
                        Vec3::splat_expr(0.0001),
                        Vec3::splat_expr(BRICK_SIZE as f32 - 0.0001),
                    ),
                    |pos, last_t, next_t| {
                        if mask.read(pos) {
                            candidate.commit((last_t + next_t) / 2.0);
                            true.expr()
                        } else {
                            false.expr()
                        }
                    },
                );
            })
            .trace()
    });

    let draw_kernel = DEVICE.create_kernel_with_options::<fn(View)>(
        KernelBuildOptions {
            name: Some("draw_kernel_4".into()),
            enable_debug_info: true,
            enable_optimization: true,
            enable_cache: true,
            enable_fast_math: false,
            ..Default::default()
        },
        &track!(|view| {
            set_block_size([8, 8, 1]);
            let pixel = dispatch_id().xy();
            let ray_dir = view.transform * (pixel.cast_f32() - view.screen_size / 2.0).extend(1.0);
            let ray_dir = ray_dir.normalize();

            let hit = trace(view.pos, ray_dir);

            if hit.procedural_hit() {
                let world_pos = view.pos + ray_dir * hit.committed_ray_t;
                let world_voxel = world_pos.floor().cast_i32();
                let delta_dist = (1.0 / ray_dir).abs();
                let side_dist = ((-ray_dir).signum() * (world_voxel.cast_f32() - world_pos)
                    + (-ray_dir).signum() * 0.5
                    + 0.5)
                    * delta_dist;
                let mask = side_dist <= keter::min(side_dist.yzx(), side_dist.zxy());
                let normal = -mask.select(ray_dir.signum().cast_i32(), Vec3::splat_expr(0));
                let prev_voxel = world_voxel + normal;
                let shadowed = trace(
                    prev_voxel.cast_f32() + 0.5,
                    Vec3::<f32>::expr(0.3, 0.2, 1.0).normalize(),
                )
                .procedural_hit();

                let model = rendered_models.read(hit.inst);
                let ray_start = model.rotation.transpose() * (view.pos - model.position);
                let ray_dir = model.rotation.transpose() * ray_dir;
                let brick = model_brick_offsets.read(model.index) + hit.prim;
                let pos = ray_start + ray_dir * hit.committed_ray_t
                    - brick_positions.read(brick).cast_f32();
                let pos = pos
                    .floor()
                    .cast_i32()
                    .clamp(Vec3::splat(0), Vec3::splat(BRICK_SIZE as i32 - 1))
                    .cast_u32();

                let color = brick_colors.read(
                    brick * (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE)
                        + pos.x
                        + pos.y * BRICK_SIZE
                        + pos.z * BRICK_SIZE * BRICK_SIZE,
                );
                // Vec3::splat_expr(num_ddas.cast_f32() * 0.1); // pcg3df(Vec3::expr(hit.inst, hit.prim, 1));
                let color = palette.read(color.cast_u32());
                app.set_pixel(
                    pixel.cast_i32(),
                    color
                        * if shadowed {
                            0.1_f32.expr()
                        } else {
                            1.0_f32.expr()
                        },
                );
            }
        }),
    );

    let mut camera = Camera {
        screen_size: app.size().map(|x| x as f32).into(),
        yaw: 0.5,
        pitch: 0.6,
        pos: Vec3::new(-500.0, -500.0, 500.0),
        fov: 1.15,
    };

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

        rt.log_fps();

        draw_kernel.dispatch(rt.dispatch_size(), &camera.view());
    });
}
