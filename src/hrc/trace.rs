use std::marker::PhantomData;

use keter::lang::types::vector::{Mat3, Vec2, Vec3};
use keter::prelude::*;

use super::color::{Color, ColorType, ToFluence};
use super::fluence::{Fluence, FluenceType, PartialTransmittance, Radiance};
use super::{Axis, Grid, Probe, quads};
use crate::utils::{HIT_STATE, LEAVE_STATE, dda, to_linear_expr};

pub trait Tracer<F: FluenceType> {
    /// Traces the upper and lower frustrums for a given probe.
    /// TODO: Fill this in
    fn trace(&self, probe: Expr<Probe>) -> [Expr<Fluence<F>>; 4];
}

pub trait WorldTracer<F: FluenceType> {
    fn trace(
        &self,
        start: Expr<Vec3<f32>>,
        direction: Expr<Vec3<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>>;
    #[tracked]
    fn trace_to(&self, start: Expr<Vec3<f32>>, end: Expr<Vec3<f32>>) -> Expr<Fluence<F>> {
        let delta = end - start;
        let length = delta.length();
        self.trace(start, delta / length, length)
    }
}

impl<F: FluenceType, T: WorldTracer<F>> WorldTracer<F> for &T {
    fn trace(
        &self,
        start: Expr<Vec3<f32>>,
        direction: Expr<Vec3<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>> {
        T::trace(*self, start, direction, length)
    }
    fn trace_to(&self, start: Expr<Vec3<f32>>, end: Expr<Vec3<f32>>) -> Expr<Fluence<F>> {
        T::trace_to(*self, start, end)
    }
}

pub struct WorldMapper<F: FluenceType, T> {
    pub tracer: T,
    pub grid: Expr<Grid>,
    pub rotation: Expr<Mat3>,
    pub offset: Expr<Vec3<f32>>,
    pub _marker: PhantomData<F>,
}
impl<F: FluenceType, T> WorldMapper<F, T> {
    #[tracked]
    pub fn to_world(&self, cell: Expr<Vec3<i32>>) -> Expr<Vec3<f32>> {
        let pos_norm = cell.cast_f32() / self.grid.size.cast_f32();
        self.rotation * pos_norm + self.offset
    }
    #[tracked]
    #[tracked]
    pub fn compute_ray(&self, probe: Expr<Probe>) -> Expr<Fluence<F>>
    where
        T: WorldTracer<F>,
    {
        let start = self.to_world(probe.cell);
        let end_cell = probe.cell + self.grid.ray_offset(probe.dir).extend(1);
        let end = self.to_world(end_cell);
        self.tracer.trace_to(start, end)
    }
}

impl<F: FluenceType, T: WorldTracer<F>> Tracer<F> for WorldMapper<F, T> {
    #[tracked]
    fn trace(&self, probe: Expr<Probe>) -> [Expr<Fluence<F>>; 4] {
        let next_grid = self.grid.next();

        let cell = probe.cell;
        let dir = probe.dir;

        let start = self.to_world(cell);
        let sizes = quads().map(|quad| next_grid.angle_size(dir * 2 + quad.expr().cast_u32()));

        let lower_offset = self.grid.lower_offset(dir);
        let upper_offset = self.grid.upper_offset(dir);

        let factor = (cell.z % 2 == 0).select(2_i32.expr(), 1_i32.expr());

        [0, 1, 2, 3].map(|i| {
            let end = self.to_world(
                cell + quads()[i]
                    .expr()
                    .select(upper_offset, lower_offset)
                    .extend(1)
                    * factor,
            );
            self.tracer.trace_to(start, end).restrict_angle(sizes[i])
        })
    }
}

pub struct VoxelTracer<C: ColorType>
where
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    pub palette: BufferView<Color<C>>,
    pub buffer: BufferView<u8>,
    pub size: Vec3<u32>,
}
impl<C: ColorType, F: FluenceType> WorldTracer<F> for VoxelTracer<C>
where
    C: ToFluence<F>,
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    #[tracked]
    fn trace(
        &self,
        start: Expr<Vec3<f32>>,
        direction: Expr<Vec3<f32>>,
        length: Expr<f32>,
    ) -> Expr<Fluence<F>> {
        let fluence = Fluence::<F>::empty().var();
        let state = dda(
            start,
            direction,
            length,
            (Vec3::splat_expr(0.01), self.size.expr().cast_f32() - 0.01),
            |pos, last_t, next_t| {
                let segment_size = next_t - last_t;
                let value = self.buffer.read(to_linear_expr(pos, self.size));
                if value != 255 {
                    let color = self.palette.read(value.cast_u32());
                    *fluence = fluence.over(color.to_fluence(segment_size));
                }
                false.expr()
            },
        );
        if state & LEAVE_STATE != 0 && state & HIT_STATE == 0 {
            *fluence = fluence.over(Fluence::solid(F::Radiance::black()).expr());
        }
        **fluence
    }
}

#[derive(Clone)]
pub struct StorageTracer<F: FluenceType> {
    pub axes: [Axis; 5],
    pub grid: Expr<Grid>,
    pub buffer: BufferVar<Fluence<F>>,
    pub next_buffer: BufferVar<Fluence<F>>,
}
impl<F: FluenceType> StorageTracer<F> {
    #[tracked]
    pub fn load(&self, probe: Expr<Probe>) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        self.buffer.read(Axis::join(
            self.axes,
            (self.grid.size, Vec2::splat_expr(self.grid.directions + 1)),
            (cell, probe.dir),
        ))
    }
    #[tracked]
    pub fn load_next(&self, probe: Expr<Probe>) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        let next_grid = self.grid.next();
        self.next_buffer.read(Axis::join(
            self.axes,
            (next_grid.size, Vec2::splat_expr(next_grid.directions + 1)),
            (cell, probe.dir),
        ))
    }
    #[tracked]
    pub fn load_opt(&self, probe: Expr<Probe>) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        if (cell >= self.grid.size).any() {
            Fluence::empty().expr()
        } else {
            self.buffer.read(Axis::join(
                self.axes,
                (self.grid.size, Vec2::splat_expr(self.grid.directions + 1)),
                (cell, probe.dir),
            ))
        }
    }
    #[tracked]
    pub fn store_next(&self, probe: Expr<Probe>, fluence: Expr<Fluence<F>>) {
        let cell = probe.cell.cast_u32();
        let next_grid = self.grid.next();
        self.next_buffer.write(
            Axis::join(
                self.axes,
                (next_grid.size, Vec2::splat_expr(next_grid.directions + 1)),
                (cell, probe.dir),
            ),
            fluence,
        );
    }
}

impl<F: FluenceType> Tracer<F> for StorageTracer<F> {
    #[tracked]
    fn trace(&self, probe: Expr<Probe>) -> [Expr<Fluence<F>>; 4] {
        // TODO: Can compress if doing bindless or something.
        let next_grid = self.grid.next();
        if probe.cell.z % 2 == 0 {
            quads().map(|quad| {
                self.load_next(
                    probe
                        .next()
                        .with_dir((probe.dir + quad.expr().cast_u32()) * 2),
                )
                .restrict_angle(next_grid.angle_size(probe.dir * 2 + quad.expr().cast_u32()))
            })
        } else {
            quads().map(|quad| {
                self.load(probe.with_dir(probe.dir + quad.expr().cast_u32()))
                    .restrict_angle(next_grid.angle_size(probe.dir * 2 + quad.expr().cast_u32()))
            })
        }
    }
}

// TODO: try to prevent integer division / multiplication
#[tracked]
pub fn merge_up<F: FluenceType>(
    storage: &StorageTracer<F>,
    next_probe: Expr<Probe>,
) -> Expr<Fluence<F>>
where
    F::Transmittance: PartialTransmittance,
{
    let next_cell = next_probe.cell;
    let next_dir = next_probe.dir;

    let next_grid = storage.grid.next();
    let cell = Vec3::expr(next_cell.x, next_cell.y, next_cell.z * 2);
    let offset = next_grid.ray_offset(next_dir);
    Fluence::blend_n(&quads().map(|quad| {
        let odd = next_dir % 2;
        let mid_offset = offset / 2;
        let dir = next_dir / 2 + odd * quad.expr().cast_u32();
        let midpoint = Probe::expr(
            cell + (mid_offset + odd.cast_i32() * (2 * quad.expr().cast_i32() - 1)).extend(1),
            next_dir / 2 + odd * (1 - quad.expr().cast_u32()),
        );
        F::over(
            storage.load(Probe::expr(cell, dir)),
            storage.load_opt(midpoint),
        )
    }))
}
