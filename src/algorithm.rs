use std::ops::AddAssign;
use std::path::PathBuf;

use keter::bindless::{BINDLESS, Tex2dHandle};
use keter::graph::NodeConfigs;
use keter::lang::types::shared::Shared;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use serde::Deserialize;
use yesod::camera::View;
use yesod::device_println;
use yesod::direction::{DirectionEncoder, EquirectangularEncoder, SphericalEncoder};
use yesod::rand::pcg3df;
use yesod::utils::{dispatch0, dispatch1, memo, offset_ray};

use crate::algorithm::rc::RcData;
use crate::cache::{Fluence, Probe, Radiance};

#[cfg(not(feature = "voxel"))]
type World = crate::world::TriangleWorld;
#[cfg(not(feature = "voxel"))]
type Hit = keter::rtx::SurfaceHit;
#[cfg(feature = "voxel")]
type World = crate::world::VoxelWorld;
#[cfg(feature = "voxel")]
type Hit = keter::rtx::CommittedHit;

// Z-up to Y-up
fn default_transform() -> glam::Mat3 {
    glam::Mat3::from_cols_array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
}

fn default_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct TriangleWorldData {
    pub path: PathBuf,
    #[serde(default = "default_transform")]
    pub transform: glam::Mat3,
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default)]
    pub material_override: Option<glam::Vec3>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorldData {
    pub world: TriangleWorldData,
    pub envmap: PathBuf,
    pub envmap_scale: f32,
    pub sun_dir: glam::Vec3,
    pub sun_color: glam::Vec3,
    #[serde(default)]
    pub custom_emitters: Vec<(u32, glam::Vec3)>,
}

pub struct CoreData {
    pub world: World,
    pub view: Singleton<View>,
    pub depth: Tex2dHandle<f32>,
    pub normal: Tex2d<Vec2<f32>>,
    pub hit: Buffer<Hit>,

    pub sun_dir: Vec3<f32>,
    pub sun_color: Vec3<f32>,
    pub environment: Tex2dHandle<Vec4<f32>>,
    pub environment_scale: f32,
    pub custom_emitters: Vec<(u32, Vec3<f32>)>,
}
impl CoreData {
    pub fn new(
        [width, height]: [u32; 2],
        WorldData {
            world,
            envmap,
            envmap_scale,
            sun_dir,
            sun_color,
            custom_emitters,
        }: WorldData,
    ) -> Self {
        #[cfg(not(feature = "voxel"))]
        let world = World::load(
            &world.path,
            world.transform * world.scale,
            world.material_override.map(Into::into),
        );
        #[cfg(feature = "voxel")]
        let world = World::load(world.path.to_str().unwrap());
        let envmap = image::open(&envmap)
            .unwrap_or_else(|e| panic!("Failed to load envmap {}: {e}", envmap.display()))
            .to_rgb32f();
        let sun_dir = sun_dir.normalize().into();
        let sun_color = sun_color.into();
        let custom_emitters = custom_emitters
            .into_iter()
            .map(|(i, color)| (i, color.into()))
            .collect();

        let depth = DEVICE.create_tex2d::<f32>(PixelStorage::Float1, width, height, 1);
        let depth = BINDLESS.push_tex2d(
            depth,
            Sampler {
                filter: SamplerFilter::LinearLinear,
                address: SamplerAddress::Edge,
            },
        );
        let environment =
            DEVICE.create_tex2d(PixelStorage::Float4, envmap.width(), envmap.height(), 1);
        let envmap = envmap
            .pixels()
            .map(|p| Vec4::new(p[0], p[1], p[2], 1.0))
            .collect::<Vec<_>>();
        environment.copy_from_vec(envmap).execute_blocking();
        let environment = BINDLESS.push_tex2d(
            environment,
            Sampler {
                filter: SamplerFilter::LinearLinear,
                address: SamplerAddress::Repeat,
            },
        );
        BINDLESS.update();
        // is the compression worth it?
        Self {
            world,
            view: Singleton::new_uninit(),
            depth,
            normal: DEVICE.create_tex2d::<Vec2<f32>>(PixelStorage::Short2, width, height, 1),
            hit: DEVICE.create_buffer::<Hit>(width as usize * height as usize),
            sun_dir,
            sun_color,
            environment,
            environment_scale: envmap_scale,
            custom_emitters,
        }
    }
    pub fn screen_size(&self) -> Vec2<u32> {
        Vec2::new(self.normal.width(), self.normal.height())
    }
    pub fn screen_dispatch_size(&self) -> [u32; 3] {
        [self.normal.width(), self.normal.height(), 1]
    }
    pub fn num_pixels(&self) -> u32 {
        self.normal.width() * self.normal.height()
    }
    #[tracked]
    pub fn get_emission(
        &self,
        rc: Option<&RcData>,
        pos: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<Radiance> {
        let normal = self.world.get_normal_flat(hit, dir);
        let weight = normal.dot(self.sun_dir);
        let radiance = Vec3::splat(0.0_f32).var();

        if weight > 0.0 {
            *radiance += self
                .world
                .trace(offset_ray(pos, normal), self.sun_dir.expr())
                .miss()
                .select(self.sun_color.expr(), Vec3::splat_expr(0.0_f32))
                * weight;
        }
        if let Some(rc) = rc {
            *radiance += rc.sample_secondary(pos, normal);
        }

        for (emitter_inst, emitter_color) in self.custom_emitters.iter() {
            if hit.inst == *emitter_inst {
                *radiance += *emitter_color;
            }
        }

        let output = radiance * self.world.get_diffuse(hit);
        output
    }
    #[tracked]
    pub fn get_radiance(
        &self,
        rc: Option<&RcData>,
        start: Expr<Vec3<f32>>,
        dir: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
    ) -> Expr<Radiance> {
        if hit.miss() {
            self.get_environment(dir)
        } else {
            self.get_emission(rc, start + dir * hit.committed_ray_t, hit, dir)
        }
    }
    #[tracked]
    pub fn get_environment(&self, dir: Expr<Vec3<f32>>) -> Expr<Radiance> {
        // Vec3::splat_expr(0.0f32)
        /*let facing = dir.dot(Vec3::z());
        let base = Vec3::splat(1.0f32);
        let falloff = Vec3::new(0.9, 0.9, 0.2);
        let downwards_falloff = 1.0 / (1.0 + 2000.0 * keter::max(-facing, 0.0));
        downwards_falloff * (base - keter::max(facing, 0.0).powf(0.1) * falloff)*/
        let uv = EquirectangularEncoder::encode(dir);
        self.environment.var().sample(uv).xyz() * self.environment_scale
    }
    #[tracked]
    pub fn get_fluence(
        &self,
        rc: Option<&RcData>,
        start: Expr<Vec3<f32>>,
        dir: Expr<Vec3<f32>>,
        hit: Expr<Hit>,
        infinite: Expr<bool>,
    ) -> Expr<Fluence> {
        if hit.miss() {
            if infinite {
                self.get_environment(dir).extend(0.0)
            } else {
                Vec3::splat_expr(0.0_f32).extend(1.0)
            }
        } else {
            self.get_emission(rc, start + dir * hit.committed_ray_t, hit, dir)
                .extend(0.0)
        }
    }
    #[tracked]
    pub fn get_world_pos(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let view = self.view.read();
        let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();
        let depth = self.depth.read(pixel);
        if depth == 0.0 {
            return;
        }
        let output = view.pos + ray_dir * depth;
        output
    }
    #[tracked]
    pub fn read_normal(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        SphericalEncoder::decode(self.normal.read(pixel))
    }
    /*#[tracked]
    pub fn read_normal_smooth(&self, pixel: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        let hit = self.hit.read(self.pixel_to_linear(pixel));
        self.world.get_normal(hit)
    }*/
    #[tracked]
    pub fn read_depth(&self, pixel: Expr<Vec2<u32>>) -> Expr<f32> {
        // could use hit.committed_ray_t instead
        self.depth.read(pixel)
    }
    #[tracked]
    pub fn pixel_to_linear(&self, pixel: Expr<Vec2<u32>>) -> Expr<u32> {
        pixel.y * self.screen_size().x + pixel.x
    }
    pub fn compute_gbuffer(&self) -> NodeConfigs<'static> {
        dispatch0(
            track!(|| {
                set_block_size([8, 8, 1]);
                let pixel = dispatch_id().xy();
                let view = self.view.read();
                let ray_dir = view.ray_dir(pixel.cast_f32() + 0.5).normalize();

                let hit = self.world.trace(view.pos, ray_dir);
                if hit.miss() {
                    self.depth.write(pixel, 0.0);
                    return;
                }
                self.depth.write(pixel, hit.committed_ray_t);
                let normal = self.world.get_normal_flat(hit, ray_dir);
                self.normal.write(pixel, SphericalEncoder::encode(normal));
                self.hit.write(self.pixel_to_linear(pixel), hit);
            }),
            self.screen_dispatch_size(),
        )
        .debug("Compute G-Buffer")
    }
}

pub mod ddgi;
pub mod rc;
pub mod ris;
pub mod splat;
pub mod trace;

#[derive(Default)]
pub struct CommandList<'a> {
    unfinished: Vec<NodeConfigs<'a>>,
    pub timings: Vec<(String, f32)>,
}
impl<'a> AddAssign<NodeConfigs<'a>> for CommandList<'a> {
    fn add_assign(&mut self, rhs: NodeConfigs<'a>) {
        self.unfinished.push(rhs);
    }
}
impl<'a> CommandList<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn execute(&mut self) {
        let commands = std::mem::take(&mut self.unfinished).chain();
        #[cfg(feature = "trace")]
        self.timings.extend(commands.execute_timed());
        #[cfg(not(feature = "trace"))]
        commands.execute();
    }
    pub fn push(&mut self, node: impl AsNodes<'a>) {
        self.unfinished.push(node.into_node_configs());
    }
}
