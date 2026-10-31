use std::f32;
use std::fmt::Debug;
use std::mem::forget;
use std::path::Path;

use dot_vox::{DotVoxData, SceneNode};
use glam::{IVec3, Mat3 as FMat3, UVec3, Vec3 as FVec3};
use itertools::Itertools;
use keter::lang::types::vector::{Mat3, Mat4, Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter::rtx::{
    Aabb, Accel, AccelBuildRequest, AccelOption, AccelTraceOptions, AccelUsageHint, CommittedHit,
    Ray, SurfaceHit, TriangleInterpolate,
};
use tobj::LoadOptions;
use yesod::shapes::dda;

use crate::utils::*;

pub struct TriangleWorld {
    accel: Accel,
    pub positions: Buffer<[f32; 3]>,
    // pub normals: Buffer<[f32; 3]>,
    pub indices: Buffer<[u32; 3]>,
    pub offsets: Buffer<[u32; 2]>,
    // per-mesh
    pub material_indices: Buffer<u32>,
    pub material_diffuse: Buffer<Vec3<f32>>,
}
impl Debug for TriangleWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriangleWorld")
            .field("positions", &self.positions)
            .field("indices", &self.indices)
            .field("offsets", &self.offsets)
            .field("material_indices", &self.material_indices)
            .field("material_diffuse", &self.material_diffuse)
            .finish_non_exhaustive()
    }
}
impl TriangleWorld {
    pub fn new(
        positions: &[[f32; 3]],
        // normals: &[[f32; 3]],
        indices: &[[u32; 3]],
        offsets: &[[u32; 2]],
        material_indices: &[u32],
        material_diffuse: &[Vec3<f32>],
    ) -> Self {
        let positions = DEVICE.create_buffer_from_slice(positions);
        // let normals = DEVICE.create_buffer_from_slice(normals);
        let indices = DEVICE.create_buffer_from_slice(indices);
        let material_indices = DEVICE.create_buffer_from_slice(material_indices);
        let material_diffuse = DEVICE.create_buffer_from_slice(material_diffuse);

        let accel = DEVICE.create_accel(AccelOption {
            hint: AccelUsageHint::FastTrace,
            allow_compaction: true,
            allow_update: false,
        });

        for (start, end) in offsets
            .iter()
            .chain(std::iter::once(&[
                indices.len() as u32,
                positions.len() as u32,
            ]))
            .tuple_windows()
        {
            let mesh = DEVICE.create_mesh(
                positions.view(start[1] as usize..end[1] as usize),
                indices.view(start[0] as usize..end[0] as usize),
                AccelOption {
                    hint: AccelUsageHint::FastTrace,
                    allow_compaction: true,
                    allow_update: false,
                },
            );
            mesh.build(AccelBuildRequest::ForceBuild);
            accel.push_mesh(&mesh, Mat4::identity(), 0xFF, true);
        }
        accel.build(AccelBuildRequest::ForceBuild);

        let offsets = DEVICE.create_buffer_from_slice(offsets);

        Self {
            accel,
            positions,
            // normals,
            indices,
            offsets,
            material_indices,
            material_diffuse,
        }
    }
    pub fn load(
        path: impl AsRef<Path> + Debug,
        transform: glam::Mat3,
        material_override: Option<Vec3<f32>>,
    ) -> Self {
        let (models, materials) = tobj::load_obj(
            path,
            &LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )
        .unwrap();
        let diffuse = if let Some(material) = material_override {
            vec![material; 1]
        } else if let Ok(materials) = materials {
            materials
                .iter()
                .map(|m| m.diffuse.unwrap_or([0.7; 3]).into())
                .collect::<Vec<_>>()
        } else {
            panic!("No materials provided");
        };

        let mut positions: Vec<[f32; 3]> = vec![];
        let mut normals: Vec<[f32; 3]> = vec![];
        let mut indices: Vec<[u32; 3]> = vec![];
        let mut offsets: Vec<[u32; 2]> = vec![];

        let mut material_indices = Vec::<u32>::new();

        for model in models {
            let mesh = model.mesh;

            // assert_eq!(mesh.normals.len() / 3, mesh.positions.len() / 3);

            offsets.push([indices.len() as u32, positions.len() as u32]);

            positions.extend(
                bytemuck::cast_slice::<_, [f32; 3]>(&mesh.positions)
                    .iter()
                    .map(|&p| <[f32; 3]>::from(transform * glam::Vec3::from(p))),
            ); // Z-up to Y-up
            //  normals.extend(bytemuck::cast_slice(&mesh.normals));
            indices.extend(bytemuck::cast_slice(&mesh.indices));

            material_indices.push(mesh.material_id.unwrap_or(0).min(diffuse.len() - 1) as u32);
        }
        Self::new(
            &positions,
            // &normals,
            &indices,
            &offsets,
            &material_indices,
            &diffuse,
        )
    }

    #[tracked]
    pub fn trace_segment(
        &self,
        ray_start: Expr<Vec3<f32>>,
        ray_dir: Expr<Vec3<f32>>,
        interval: Expr<Vec2<f32>>,
    ) -> Expr<SurfaceHit> {
        self.accel.intersect(
            Ray::new_expr(
                Expr::<[f32; 3]>::from(ray_start),
                interval.x,
                Expr::<[f32; 3]>::from(ray_dir),
                interval.y,
            ),
            AccelTraceOptions {
                mask: 0xFF.expr(),
                ..Default::default()
            },
        )
    }

    #[tracked]
    pub fn trace(&self, ray_start: Expr<Vec3<f32>>, ray_dir: Expr<Vec3<f32>>) -> Expr<SurfaceHit> {
        self.trace_segment(ray_start, ray_dir, Vec2::expr(0.0, f32::INFINITY))
    }

    #[tracked]
    pub fn get_normal_oriented(&self, hit: Expr<SurfaceHit>) -> Expr<Vec3<f32>> {
        let offsets = self.offsets.read(hit.inst);
        let triangle = self.indices.read(offsets[0] + hit.prim);
        let positions: [_; 3] = std::array::from_fn(|i| {
            Expr::<Vec3<f32>>::from(self.positions.read(triangle[i as u32] + offsets[1]))
        });
        (positions[1] - positions[0])
            .cross(positions[2] - positions[0])
            .normalize()
    }

    #[tracked]
    pub fn get_normal_flat(&self, hit: Expr<SurfaceHit>, dir: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
        let normal = self.get_normal_oriented(hit);
        -normal.dot(dir).signum() * normal
    }

    /*
    #[tracked]
    pub fn get_normal(&self, hit: Expr<SurfaceHit>) -> Expr<Vec3<f32>> {
        let offsets = self.offsets.read(hit.inst);
        let triangle = self.indices.read(offsets[0] + hit.prim);
        let normals: [_; 3] = std::array::from_fn(|i| {
            Expr::<Vec3<f32>>::from(self.normals.read(triangle[i as u32] + offsets[1]))
        });
        hit.interpolate(normals[0], normals[1], normals[2])
            .normalize()
    }
    */

    #[tracked]
    pub fn get_diffuse(&self, hit: Expr<SurfaceHit>) -> Expr<Vec3<f32>> {
        let material_index = self.material_indices.read(hit.inst);
        self.material_diffuse.read(material_index)
    }
}

pub struct VoxelWorld {
    // bindless: BindlessArray,
    accel: Accel,
    rendered_models: Buffer<RenderedModel>,
    model_brick_offsets: Buffer<u32>,
    brick_masks: Buffer<BrickMask>,
    brick_positions: Buffer<Vec3<u32>>,
    brick_colors: Buffer<u8>,
    palette: Buffer<Vec3<f32>>,
}

impl VoxelWorld {
    pub fn load(path: &str) -> Self {
        let data = dot_vox::load(path).unwrap();
        Self::new(data)
    }
    pub fn new(data: DotVoxData) -> Self {
        // let bindless = DEVICE.create_bindless_array(data.models.len());

        let mut brick_masks = Vec::<BrickMask>::new();
        let mut brick_colors = Vec::<u8>::new();
        let mut brick_positions = Vec::<Vec3<u32>>::new();
        let mut model_bvhs = Vec::new();
        let mut model_sizes = Vec::<Vec3<u32>>::new();
        let mut model_brick_offsets = Vec::<u32>::new();
        for (index, model) in data.models.iter().enumerate() {
            let s = model.size;
            let s = UVec3::new(s.x, s.y, s.z);
            let bricks = s.map(|x| x.div_ceil(BRICK_SIZE));
            let s = bricks * BRICK_SIZE;
            model_brick_offsets.push(brick_masks.len() as u32);
            let mut aabbs = Vec::<Aabb>::new();
            // let mut brick_map = vec![u32::MAX; bricks.element_product() as usize];
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
                    // brick_map[to_linear(brick, bricks)] = brick_masks.len() as u32;
                    brick_masks.push(mask);
                    brick_colors.extend_from_slice(&colors);
                    brick_positions.push((brick * BRICK_SIZE).into());
                    aabbs.push(Aabb {
                        min: (brick * BRICK_SIZE + min).as_vec3().into(),
                        max: (brick * BRICK_SIZE + max).as_vec3().into(),
                    });
                }
            }
            // let brick_map = DEVICE.create_buffer_from_slice(&brick_map);
            // bindless.emplace_buffer(index, &brick_map);
            // forget(brick_map);
            if aabbs.is_empty() {
                println!("Voxels: {:?}", model.voxels);
                println!("Bricks: {bricks:?}");
                println!("Size: {s:?}");
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
            if index.is_multiple_of(20) {
                println!("Loading: {index:?}");
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
            &data,
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

        let palette = DEVICE.create_buffer_from_fn(data.palette.len(), |i| {
            let color = data.palette[i];
            Vec3::new(color.r as f32, color.g as f32, color.b as f32).map(|x| (x / 255.0).powf(2.2))
        });

        Self {
            // bindless,
            accel,
            rendered_models,
            model_brick_offsets,
            brick_masks,
            brick_positions,
            brick_colors,
            palette,
        }
    }

    #[tracked]
    pub fn trace_segment(
        &self,
        ray_start: Expr<Vec3<f32>>,
        ray_dir: Expr<Vec3<f32>>,
        interval: Expr<Vec2<f32>>,
    ) -> Expr<CommittedHit> {
        // TODO: should change to fit with other api
        self.accel
            .traverse(
                Ray::new_expr(
                    Expr::<[f32; 3]>::from(ray_start),
                    interval.x,
                    Expr::<[f32; 3]>::from(ray_dir),
                    interval.y,
                ),
                AccelTraceOptions {
                    mask: 0xFF.expr(),
                    ..Default::default()
                },
            )
            .on_procedural_hit(|candidate| {
                let model = self.rendered_models.read(candidate.inst);
                let brick = self.model_brick_offsets.read(model.index) + candidate.prim;
                let mask = self.brick_masks.read(brick);

                let ray_start = model.rotation.transpose() * (ray_start - model.position)
                    - self.brick_positions.read(brick).cast_f32();
                let ray_dir = model.rotation.transpose() * ray_dir;

                dda(
                    ray_start,
                    ray_dir,
                    10000.0.expr(),
                    (
                        Vec3::splat_expr(0.0001),
                        Vec3::splat_expr(BRICK_SIZE as f32 - 0.0001),
                    ),
                    |pos, last_t, _next_t| {
                        if mask.read(pos.cast_u32()) {
                            // TODO: Commit the actual position instead of this stupid hack
                            candidate.commit(last_t); // (last_t + next_t) / 2.0);
                            true.expr()
                        } else {
                            false.expr()
                        }
                    },
                );
            })
            .trace()
    }

    #[tracked]
    pub fn trace(
        &self,
        ray_start: Expr<Vec3<f32>>,
        ray_dir: Expr<Vec3<f32>>,
    ) -> Expr<CommittedHit> {
        self.trace_segment(ray_start, ray_dir, Vec2::expr(0.0, f32::INFINITY))
    }

    #[tracked]
    pub fn get_normal(&self, ray_dir: Expr<Vec3<f32>>, pos: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
        let nearest = pos.round();
        let diff = (pos - nearest).abs();
        let mask = diff <= keter::min(diff.yzx(), diff.zxy());
        mask.select(Vec3::splat_expr(-1.0), Vec3::splat_expr(0.0)) * ray_dir.signum()
    }

    #[tracked]
    pub fn get_voxel(&self, pos: Expr<Vec3<f32>>, normal: Expr<Vec3<f32>>) -> Expr<Vec3<i32>> {
        (pos - 0.5 * normal).floor().cast_i32()
    }

    #[tracked]
    pub fn get_color(
        &self,
        ray_start: Expr<Vec3<f32>>,
        ray_dir: Expr<Vec3<f32>>,
        hit: Expr<CommittedHit>,
    ) -> Expr<Vec3<f32>> {
        self.palette.read(
            self.brick_colors
                .read(self.get_local(ray_start, ray_dir, hit))
                .cast_u32(),
        )
    }

    #[tracked]
    pub fn get_local(
        &self,
        ray_start: Expr<Vec3<f32>>,
        ray_dir: Expr<Vec3<f32>>,
        hit: Expr<CommittedHit>,
    ) -> Expr<u32> {
        let model = self.rendered_models.read(hit.inst);
        let ray_start = model.rotation.transpose() * (ray_start - model.position);
        let ray_dir = model.rotation.transpose() * ray_dir;
        let brick = self.model_brick_offsets.read(model.index) + hit.prim;
        let pos =
            ray_start + ray_dir * hit.committed_ray_t - self.brick_positions.read(brick).cast_f32();
        let pos = self.get_voxel(pos, self.get_normal(ray_dir, pos));
        let pos = pos
            .clamp(Vec3::splat(0), Vec3::splat(BRICK_SIZE as i32 - 1))
            .cast_u32();

        brick * (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE)
            + pos.x
            + pos.y * BRICK_SIZE
            + pos.z * BRICK_SIZE * BRICK_SIZE
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct RenderedModel {
    pub position: Vec3<f32>,
    pub rotation: Mat3,
    pub index: u32,
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

pub const BRICK_SIZE: u32 = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
pub struct BrickMask {
    pub values: u64,
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
