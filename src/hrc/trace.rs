use std::marker::PhantomData;

use keter::lang::types::vector::{Mat3, Vec3};
use keter::prelude::*;

use super::color::{Color, ColorType, ToFluence};
use super::fluence::{Fluence, FluenceType};
use super::{Grid, Probe, quads};
use crate::utils::{dda, to_linear_expr};

pub trait Tracer<F: FluenceType> {
    /// Traces the upper and lower frustrums for a given probe.
    /// TODO: Fill this in
    fn trace(&self, grid: Expr<Grid>, probe: Expr<Probe>) -> [Expr<Fluence<F>>; 4];
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

pub struct WorldMapper<F: FluenceType, T> {
    pub tracer: T,
    pub rotation: Mat3,
    pub offset: Vec3<f32>,
    pub _marker: PhantomData<F>,
}
impl<F: FluenceType, T> WorldMapper<F, T> {
    #[tracked]
    pub fn to_world(&self, grid: Expr<Grid>, cell: Expr<Vec3<i32>>) -> Expr<Vec3<f32>> {
        let pos_norm = cell.cast_f32() / grid.size.cast_f32();
        self.rotation.expr() * pos_norm + self.offset
    }
}

impl<F: FluenceType, T: WorldTracer<F>> Tracer<F> for WorldMapper<F, T> {
    #[tracked]
    fn trace(&self, grid: Expr<Grid>, probe: Expr<Probe>) -> [Expr<Fluence<F>>; 4] {
        let next_grid = grid.next();

        let cell = probe.cell;
        let dir = probe.dir;

        let start = self.to_world(grid, cell);
        let sizes = quads().map(|quad| next_grid.angle_size(dir * 2 + quad.expr().cast_u32()));

        let lower_offset = grid.lower_offset(dir);
        let upper_offset = grid.upper_offset(dir);

        let factor = (cell.x % 2 == 0).select(2_i32.expr(), 1_i32.expr());

        [0, 1, 2, 3].map(|i| {
            self.tracer
                .trace_to(
                    start,
                    self.to_world(
                        grid,
                        cell + quads()[i]
                            .expr()
                            .select(lower_offset, upper_offset)
                            .extend(1)
                            * factor,
                    ),
                )
                .restrict_angle(sizes[i])
        })
    }
}

pub struct VoxelTracer<C: ColorType>
where
    C::Emission: IoTexel,
    C::Opacity: IoTexel,
{
    palette: BufferView<Color<C>>,
    buffer: BufferView<u8>,
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
        dda(
            start,
            direction,
            length,
            (Vec3::splat_expr(0.001), self.size.expr().cast_f32() - 0.001),
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
        **fluence
    }
}
