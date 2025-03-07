use fluence::{FluenceType, Radiance};
use keter::lang::types::vector::{Vec2, Vec3};
use keter::prelude::*;
use storage::RadianceStorage;
use trace::Tracer;

pub mod color;
pub mod fluence;
pub mod storage;
pub mod trace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    CellX,
    CellY,
    CellZ,
    DirectionU,
    DirectionV,
}
impl Axis {
    pub fn get(self, (cell, dir): (Expr<Vec3<u32>>, Expr<Vec2<u32>>)) -> Expr<u32> {
        match self {
            Axis::CellX => cell.x,
            Axis::CellY => cell.y,
            Axis::CellZ => cell.z,
            Axis::DirectionU => dir.x,
            Axis::DirectionV => dir.y,
        }
    }
    pub fn get_mut(self, (cell, dir): (Var<Vec3<u32>>, Var<Vec2<u32>>)) -> Var<u32> {
        match self {
            Axis::CellX => cell.x,
            Axis::CellY => cell.y,
            Axis::CellZ => cell.z,
            Axis::DirectionU => dir.x,
            Axis::DirectionV => dir.y,
        }
    }
    #[tracked]
    pub fn join(
        mut axes: [Self; 5],
        size: (Expr<Vec3<u32>>, Expr<Vec2<u32>>),
        probe: (Expr<Vec3<u32>>, Expr<Vec2<u32>>),
    ) -> Expr<u32> {
        axes.reverse();
        let mut result = 0_u32.expr();
        for &axis in &axes {
            result = result * axis.get(size) + axis.get(probe);
        }
        result
    }
    #[tracked]
    pub fn extract(
        axes: [Self; 5],
        size: (Expr<Vec3<u32>>, Expr<Vec2<u32>>),
        mut value: Expr<u32>,
    ) -> (Expr<Vec3<u32>>, Expr<Vec2<u32>>) {
        let probe = (Vec3::splat(0_u32).var(), Vec2::splat(0_u32).var());

        for &axis in &axes {
            let s = axis.get(size);
            *axis.get_mut(probe) = value % s;
            value = value / axis.get(size);
        }

        (**probe.0, **probe.1)
    }
}

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct Grid {
    pub size: Vec3<u32>,
    pub directions: u32,
}
impl Grid {
    pub fn new(size: Vec3<u32>, directions: u32) -> Grid {
        Grid { size, directions }
    }
    pub fn shift(&self, n: u32) -> Grid {
        Grid {
            size: Vec3::new(self.size.x, self.size.y, self.size.z >> n),
            directions: self.directions << n,
        }
    }
    pub fn size(&self) -> u32 {
        self.size.x * self.size.y * self.size.z * self.directions * self.directions
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
            Vec3::expr(self.size.x, self.size.y, self.size.z / 2),
            self.directions * 2,
        )
    }
    #[tracked]
    pub fn last(self) -> Expr<Grid> {
        Grid::expr(
            Vec3::expr(self.size.x, self.size.y, self.size.z * 2),
            self.directions / 2,
        )
    }

    #[tracked]
    pub fn ray_offset(self, dir: Expr<Vec2<u32>>) -> Expr<Vec2<i32>> {
        2 * dir.cast_i32() - self.directions.cast_i32()
    }
    #[tracked]
    pub fn lower_offset(self, dir: Expr<Vec2<u32>>) -> Expr<Vec2<i32>> {
        2 * dir.cast_i32() - self.directions.cast_i32()
    }
    #[tracked]
    pub fn upper_offset(self, dir: Expr<Vec2<u32>>) -> Expr<Vec2<i32>> {
        2 * dir.cast_i32() - self.directions.cast_i32() + 2
    }
    #[tracked]
    pub fn offset(self, dir: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
        2.0 * dir - self.directions.cast_f32() + 1.0
    }
    #[tracked]
    pub fn angle_size(self, dir: Expr<Vec2<u32>>) -> Expr<f32> {
        // https://vixra.org/pdf/2001.0603v2.pdf
        fn area_to(size: Expr<Vec2<f32>>) -> Expr<f32> {
            ((size.x * size.y) / (1.0 + size.x.sqr() + size.y.sqr()).sqrt()).atan()
        }

        let upper = self.offset(dir.cast_f32() + 0.5) / self.directions.cast_f32();
        let lower = self.offset(dir.cast_f32() - 0.5) / self.directions.cast_f32();

        area_to(upper)
            - area_to(Vec2::expr(upper.x, lower.y))
            - area_to(Vec2::expr(lower.x, upper.y))
            + area_to(lower)
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
            Vec3::expr(self.cell.x, self.cell.y, self.cell.z / 2),
            self.dir,
        )
    }
    #[tracked]
    pub fn with_dir(self, dir: Expr<Vec2<u32>>) -> Expr<Probe> {
        Probe::expr(self.cell, dir)
    }
}

fn quads() -> [Vec2<bool>; 4] {
    [
        Vec2::new(false, false),
        Vec2::new(true, false),
        Vec2::new(false, true),
        Vec2::new(true, true),
    ]
}

#[tracked]
pub fn merge<F: FluenceType, S: RadianceStorage<F::Radiance>, T: Tracer<F>>(
    grid: Expr<Grid>,
    probe: Expr<Probe>,
    storage: &S,
    tracer: &T,
) -> Expr<F::Radiance> {
    let next_grid = grid.next();
    let load_next = |probe: Expr<Probe>| storage.load(next_grid, probe.next());

    let cell = probe.cell;
    let dir = probe.dir;

    let lower_dir = dir * 2;
    let upper_dir = dir * 2 + 1;
    let lower_offset = grid.lower_offset(dir);
    let upper_offset = grid.upper_offset(dir);

    let factor = (cell.z % 2 == 0).select(2_i32.expr(), 1_i32.expr());

    let trace = tracer.trace(grid, probe);

    let next = [0, 1, 2, 3].map(|i| {
        F::over_radiance(
            trace[i],
            load_next(Probe::expr(
                cell + quads()[i]
                    .expr()
                    .select(lower_offset, upper_offset)
                    .extend(1)
                    * factor,
                quads()[i].expr().select(lower_dir, upper_dir),
            )),
        )
    });

    if cell.z % 2 == 0 {
        F::Radiance::merge([0, 1, 2, 3].map(|i| {
            F::Radiance::blend(
                load_next(Probe::expr(
                    cell,
                    quads()[i].expr().select(lower_dir, upper_dir),
                )),
                next[i],
            )
        }))
    } else {
        F::Radiance::merge(next)
    }
}
