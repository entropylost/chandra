use std::marker::PhantomData;

use keter::lang::types::vector::{Mat3, Vec3};
use keter::prelude::*;

use super::color::{Color, ColorType, ToFluence};
use super::fluence::{Fluence, FluenceType, Radiance};
use super::{Grid, Probe, quads};
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
                    .select(lower_offset, upper_offset)
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

/*
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageTracer {
    pub axes: [Axis; 5],
    pub buffer: BufferVar<Fluence<F>>,
    pub next_buffer: BufferVar<Fluence<F>>,
}
impl StorageTracer {
    #[tracked]
    pub fn load<F: FluenceType>(
        &self,
        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        buffer.read(Axis::join(
            self.axes,
            (cell, probe.dir),
            (grid.size, grid.directions + 1),
        ))
    }
    #[tracked]
    pub fn load_opt<F: FluenceType>(
        &self,

        buffer: &BufferVar<Fluence<F>>,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> Expr<Fluence<F>> {
        let cell = probe.cell.cast_u32();
        if (cell >= grid.size).any() {
            Fluence::empty().expr()
        } else {
            buffer.read(Axis::join(
                self.axes,
                (cell, probe.dir),
                (grid.size, grid.directions + 1),
            ))
        }
    }
    #[tracked]
    pub fn store_next<F: FluenceType>(
        &self,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
        fluence: Expr<Fluence<F>>,
    ) {
        let cell = probe.cell.cast_u32();
        self.next_buffer.write(
            Axis::join(
                self.axes,
                (cell, probe.dir),
                (grid.size, grid.directions + 1),
            ),
            fluence,
        );
    }
}

impl<F: FluenceType> Tracer<F> for StorageTracer {
    type Params = (BufferVar<Fluence<F>>, BufferVar<Fluence<F>>);
    #[tracked]
    fn trace(
        &self,
        (buffer, next_buffer): &Self::Params,
        grid: Expr<Grid>,
        probe: Expr<Probe>,
    ) -> [Expr<Fluence<F>>; 2] {
        // TODO: Can compress if doing bindless or something.
        let next_grid = grid.next();
        let lower_size = next_grid.angle_size(probe.dir * 2);
        let upper_size = next_grid.angle_size(probe.dir * 2 + 1);
        if probe.cell.x % 2 == 0 {
            [
                self.load(next_buffer, next_grid, probe.next().with_dir(probe.dir * 2))
                    .restrict_angle(lower_size),
                self.load(
                    next_buffer,
                    next_grid,
                    probe.next().with_dir((probe.dir + 1) * 2),
                )
                .restrict_angle(upper_size),
            ]
        } else {
            let a = buffer.len_expr_u32() >= (grid.size.x * grid.size.y * (grid.directions + 1));
            lc_assert!(a);
            [
                self.load(buffer, grid, probe).restrict_angle(lower_size),
                self.load(buffer, grid, probe.with_dir(probe.dir + 1))
                    .restrict_angle(upper_size),
            ]
        }
    }
}

// TODO: try to prevent integer division / multiplication
#[tracked]
pub fn merge_up<F: FluenceType>(
    storage: &StorageTracer,
    last_buffer: &BufferVar<Fluence<F>>,
    grid: Expr<Grid>,
    probe: Expr<Probe>,
) -> Expr<Fluence<F>>
where
    F::Transmittance: PartialTransmittance,
{
    let cell = probe.cell;
    let dir = probe.dir;

    let last_grid = grid.last();
    let last_cell = Vec2::expr(cell.x * 2, cell.y);
    let offset = grid.ray_offset(dir);
    if dir % 2 == 0 {
        let last_dir = dir / 2;
        let midpoint = Probe::expr(
            // TODO: integer division ow
            last_cell + Vec2::expr(1, offset / 2),
            last_dir,
        );

        F::over(
            storage.load(last_buffer, last_grid, Probe::expr(last_cell, last_dir)),
            storage.load_opt(last_buffer, last_grid, midpoint),
        )
    } else {
        let last_dir_lower = dir / 2;
        let last_dir_upper = dir / 2 + 1;
        // TODO: Can simplify by expanding offset.
        let midpoint_lower = Probe::expr(
            last_cell + Vec2::expr(1, (offset.cast_f32() / 2.0).floor().cast_i32()),
            last_dir_upper,
        );
        let midpoint_upper = Probe::expr(
            last_cell + Vec2::expr(1, (offset.cast_f32() / 2.0).ceil().cast_i32()),
            last_dir_lower,
        );

        Fluence::blend(
            F::over(
                storage.load(
                    last_buffer,
                    last_grid,
                    Probe::expr(last_cell, last_dir_lower),
                ),
                storage.load_opt(last_buffer, last_grid, midpoint_lower),
            ),
            F::over(
                storage.load(
                    last_buffer,
                    last_grid,
                    Probe::expr(last_cell, last_dir_upper),
                ),
                storage.load_opt(last_buffer, last_grid, midpoint_upper),
            ),
        )
    }
}
 */
