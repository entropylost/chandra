use keter::lang::types::vector::Vec2;
use keter::prelude::*;

use super::fluence::Radiance;
use super::{Axis, Grid, Probe};

// Note: Integer Multiplication is multiple instructions on compute capacity <= 6.2 (10-series and less).

// Construct from kernel arg somehow.
// Then later there's like a Pyraimd::construct_storages() and then Pyramid[level] passin.
pub trait RadianceStorage<R: Radiance> {
    fn load(&self, grid: Expr<Grid>, probe: Expr<Probe>) -> Expr<R>;
    fn store(&self, grid: Expr<Grid>, probe: Expr<Probe>, radiance: Expr<R>);
}

pub struct BufferStorage<R: Radiance> {
    pub axes: [Axis; 5],
    pub buffer: BufferVar<R>,
}

impl<R: Radiance> RadianceStorage<R> for BufferStorage<R> {
    #[tracked]
    fn load(&self, grid: Expr<Grid>, probe: Expr<Probe>) -> Expr<R> {
        let cell = probe.cell.cast_u32();
        if (cell >= grid.size).any() {
            R::black().expr()
        } else {
            self.buffer.read(Axis::join(
                self.axes,
                (grid.size, Vec2::splat_expr(grid.directions)),
                (cell, probe.dir),
            ))
        }
    }
    #[tracked]
    fn store(&self, grid: Expr<Grid>, probe: Expr<Probe>, radiance: Expr<R>) {
        let cell = probe.cell.cast_u32();

        self.buffer.write(
            Axis::join(
                self.axes,
                (grid.size, Vec2::splat_expr(grid.directions)),
                (cell, probe.dir),
            ),
            radiance,
        );
    }
}
