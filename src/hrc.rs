use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;

mod color;
mod fluence;

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
struct Grid {
    size: Vec3<u32>,
    directions: u32,
}
impl Grid {
    pub fn new(size: Vec3<u32>, directions: u32) -> Grid {
        Grid { size, directions }
    }
    #[tracked]
    pub fn expr(size: Expr<Vec3<u32>>, directions: Expr<u32>) -> Expr<Grid> {
        Grid::from_comps_expr(GridComps { size, directions })
    }
}
impl GridExpr {
    #[tracked]
    pub fn next(self) -> Expr<Grid> {
        Grid::expr(
            Vec3::expr(self.size.x / 2, self.size.y, self.size.z),
            self.directions * 2,
        )
    }
    #[tracked]
    pub fn last(self) -> Expr<Grid> {
        Grid::expr(
            Vec3::expr(self.size.x * 2, self.size.y, self.size.z),
            self.directions / 2,
        )
    }

    #[tracked]
    pub fn ray_offset(self, dir: Expr<u32>) -> Expr<i32> {
        2 * dir.cast_i32() - self.directions.cast_i32()
    }
    #[tracked]
    pub fn lower_offset(self, dir: Expr<u32>) -> Expr<i32> {
        2 * dir.cast_i32() - self.directions.cast_i32()
    }
    #[tracked]
    pub fn upper_offset(self, dir: Expr<u32>) -> Expr<i32> {
        2 * dir.cast_i32() - self.directions.cast_i32() + 2
    }
    #[tracked]
    pub fn offset(self, dir: Expr<f32>) -> Expr<f32> {
        2.0 * dir - self.directions.cast_f32() + 1.0
    }
    #[tracked]
    pub fn ray_angle(self, dir: Expr<f32>) -> Expr<f32> {
        // Total angle is PI / 2 (assumed)
        self.offset(dir).atan2(self.directions.cast_f32())
    }
    #[tracked]
    pub fn angle_size(self, dir: Expr<u32>) -> Expr<f32> {
        let upper = self.ray_angle(dir.cast_f32() + 0.5);
        let lower = self.ray_angle(dir.cast_f32() - 0.5);
        upper - lower
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Probe {
    pub cell: Vec3<i32>,
    pub dir: Vec2<u32>,
}
impl Probe {
    pub fn expr(cell: Expr<Vec3<i32>>, dir: Expr<Vec2<u32>>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps { cell, dir })
    }
}
impl ProbeExpr {
    #[tracked]
    pub fn next(self) -> Expr<Probe> {
        Probe::expr(
            Vec3::expr(self.cell.x / 2, self.cell.y, self.cell.z),
            self.dir,
        )
    }
    #[tracked]
    pub fn with_dir(self, dir: Expr<Vec2<u32>>) -> Expr<Probe> {
        Probe::expr(self.cell, dir)
    }
}
