use glam::UVec3;
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;

#[tracked]
pub fn aabb_intersect(
    start: Expr<Vec3<f32>>,
    inv_dir: Expr<Vec3<f32>>,
    aabb_min: Expr<Vec3<f32>>,
    aabb_max: Expr<Vec3<f32>>,
) -> Expr<Vec2<f32>> {
    let t0 = (aabb_min - start) * inv_dir;
    let t1 = (aabb_max - start) * inv_dir;
    let tmin = keter::min(t0, t1).reduce_max();
    let tmax = keter::max(t0, t1).reduce_min();
    Vec2::expr(tmin, tmax)
}

#[tracked]
pub fn dda(
    start: Expr<Vec3<f32>>,
    ray_dir: Expr<Vec3<f32>>,
    length: Expr<f32>,
    bounds: (Expr<Vec3<f32>>, Expr<Vec3<f32>>),
    f: impl Fn(Expr<Vec3<u32>>, Expr<f32>, Expr<f32>) -> Expr<bool>,
) {
    let inv_dir = (ray_dir + f32::EPSILON).recip();
    let interval = aabb_intersect(start, inv_dir, bounds.0, bounds.1);
    let start_t = keter::max(interval.x, 0.0);
    let ray_start = start + start_t * ray_dir;
    let end_t = keter::min(interval.y, length);
    if end_t - start_t >= 0.01 {
        let pos = ray_start.floor().cast_i32().cast_u32().var();

        let delta_dist = inv_dir.abs();

        let ray_step = ray_dir.signum().cast_i32().cast_u32();
        let side_dist = (ray_dir.signum() * (pos.cast_i32().cast_f32() - ray_start)
            + ray_dir.signum() * 0.5
            + 0.5)
            * delta_dist;
        let side_dist = side_dist.var();

        let last_t = start_t.var();

        loop {
            let next_t = side_dist.reduce_min() + start_t;

            if f(**pos, **last_t, keter::min(next_t, end_t)) {
                break;
            }
            if next_t >= end_t {
                break;
            }
            *last_t = next_t;
            let mask = side_dist <= keter::min(side_dist.yzx(), side_dist.zxy());

            *side_dist += mask.select(delta_dist, Vec3::splat_expr(0.0));
            *pos += mask.select(ray_step, Vec3::splat_expr(0));
        }
    }
}

// https://github.com/markjarzynski/PCG3D/blob/master/pcg3d.hlsl
#[tracked]
pub fn pcg3d(v: Expr<Vec3<u32>>) -> Expr<Vec3<u32>> {
    let v = v.var();
    *v = v * 1664525u32 + 1013904223u32;

    *v.x += v.y * v.z;
    *v.y += v.z * v.x;
    *v.z += v.x * v.y;

    *v ^= v >> 16u32;

    *v.x += v.y * v.z;
    *v.y += v.z * v.x;
    *v.z += v.x * v.y;

    **v
}

#[tracked]
pub fn pcg3df(v: Expr<Vec3<u32>>) -> Expr<Vec3<f32>> {
    pcg3d(v).cast_f32() / u32::MAX as f32
}

pub fn to_linear(pos: UVec3, size: UVec3) -> usize {
    (pos.x + pos.y * size.x + pos.z * size.x * size.y) as usize
}

#[tracked]
pub fn to_linear_expr(pos: Expr<Vec3<u32>>, size: impl AsExpr<Value = Vec3<u32>>) -> Expr<u32> {
    let size = size.as_expr();
    pos.x + pos.y * size.x + pos.z * size.x * size.y
}

pub fn iter_vec(size: UVec3) -> impl Iterator<Item = UVec3> {
    (0..size.x)
        .flat_map(move |x| (0..size.y).map(move |y| (x, y)))
        .flat_map(move |(x, y)| (0..size.z).map(move |z| UVec3::new(x, y, z)))
}
