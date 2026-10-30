#![allow(unused_parens)]

use keter::graph::NodeConfigs;
use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use yesod::camera::View;
use yesod::direction::{DirectionEncoder, SphericalEncoder};
use yesod::interpolate::{Bilinear, Trilinear, TrilinearComps};
use yesod::utils::{FetchAddVector, decode_morton2_16, encode_morton2_16, next_float_down_pos_2};

use crate::table::{HashTable, INVALID_KEY, Key};
use crate::utils::{corner_to_index, index_to_corner};

const NUM_BITS: Key = 18;
const MASK: Key = (1 << NUM_BITS) - 1;

const OFFSET: u32 = 1 << 17;

pub type Radiance = Vec3<f32>;
pub type Fluence = Vec4<f32>;

#[tracked]
fn encode_position(cell: Expr<Vec4<u32>>) -> Expr<Key> {
    let cell = cell.cast_u64();
    let cell = cell & MASK;
    cell.w << (NUM_BITS * 3) | (cell.z << (NUM_BITS * 2)) | (cell.y << NUM_BITS) | cell.x
}

#[tracked]
fn decode_position(pos: Expr<Key>) -> Expr<Vec4<u32>> {
    let result = Vec4::expr(
        pos & MASK,
        (pos >> NUM_BITS) & MASK,
        (pos >> (NUM_BITS * 2)) & MASK,
        pos >> (NUM_BITS * 3),
    );
    let result = result.cast_u32();
    result
}

pub type Encoder = SphericalEncoder;
pub const INDEX_SCALING: Vec2<u32> = Vec2::new(2, 1);

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct DirectionIndexer {
    pub resolution: u32,
}
impl DirectionIndexerExpr {
    #[tracked]
    pub fn grid_to_dir_f32(&self, index: Expr<Vec2<f32>>) -> Expr<Vec3<f32>> {
        let v = index / self.grid_size().cast_f32();
        Encoder::decode(v)
    }
    #[tracked]
    pub fn grid_to_dir(&self, index: Expr<Vec2<u32>>) -> Expr<Vec3<f32>> {
        self.grid_to_dir_f32(index.cast_f32() + 0.5)
    }
    #[tracked]
    pub fn dir_to_grid(&self, dir: Expr<Vec3<f32>>) -> Expr<Vec2<u32>> {
        let v = Encoder::encode(dir);
        self.quantize(v)
    }
    #[tracked]
    pub fn dir_to_grid_bilinear(&self, dir: Expr<Vec3<f32>>) -> Expr<Bilinear> {
        let v = Encoder::encode(dir);
        Bilinear::new(v * next_float_down_pos_2(self.grid_size().cast_f32()))
    }
    #[tracked]
    pub fn quantize(&self, encoded_dir: Expr<Vec2<f32>>) -> Expr<Vec2<u32>> {
        (encoded_dir * next_float_down_pos_2(self.grid_size().cast_f32())).cast_u32()
    }
    #[tracked]
    pub fn grid_to_linear(&self, index: Expr<Vec2<u32>>) -> Expr<u32> {
        let index = index % self.grid_size();
        encode_morton2_16(index)
        // index.x * self.grid_size().y + index.y
    }
    #[tracked]
    pub fn linear_to_grid(&self, index: Expr<u32>) -> Expr<Vec2<u32>> {
        decode_morton2_16(index)
        // Vec2::expr(index / self.grid_size().y, index % self.grid_size().y)
    }
    #[tracked]
    pub fn linear_to_dir(&self, index: Expr<u32>) -> Expr<Vec3<f32>> {
        self.grid_to_dir(self.linear_to_grid(index))
    }
    #[tracked]
    pub fn dir_to_linear(&self, dir: Expr<Vec3<f32>>) -> Expr<u32> {
        self.grid_to_linear(self.dir_to_grid(dir))
    }
    #[tracked]
    pub fn linear_size(&self) -> Expr<u32> {
        self.grid_size().reduce_prod()
    }
    #[tracked]
    pub fn grid_size(&self) -> Expr<Vec2<u32>> {
        self.resolution * INDEX_SCALING
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct LoadedNode {
    pub probe: Probe,
    // position in data buffer
    // 0 if not loaded so that things don't go out of bounds
    pub data_offset: u32,
    // index in the table
    // if u32::MAX, this isn't loaded
    pub index: u32,
}
impl LoadedNodeExpr {
    #[tracked]
    pub fn invalid(&self) -> Expr<bool> {
        self.index == u32::MAX.expr()
    }
    #[tracked]
    pub fn loaded(&self) -> Expr<bool> {
        self.index != u32::MAX.expr()
    }
}

/*
Values are offset by 1 << 31 to avoid negative offsets.
So a probe at 0 would have position 1<<31, regardless of the cascade.
*/
#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
pub struct Probe {
    pub position: Vec4<u32>,
    pub cascade: u32,
}
impl Probe {
    #[tracked]
    pub fn base(pos: Expr<Vec4<u32>>) -> Expr<Self> {
        Self::from_comps_expr(ProbeComps {
            position: pos,
            cascade: 0_u32.expr(),
        })
    }
}
impl ProbeExpr {
    #[tracked]
    pub fn next_trilinear(&self, i: Expr<u32>) -> (Expr<Probe>, Expr<f32>) {
        let j = corner_to_index(!(self.position.xyz() | !1_u32));
        let base_probe = Probe::from_comps_expr(ProbeComps {
            position: (self.position.xyz() - 1).extend(self.position.w),
            cascade: self.cascade,
        })
        .next_nearest();
        let pos = base_probe.position.xyz() + index_to_corner(i);
        let next_probe = Probe::from_comps_expr(ProbeComps {
            position: pos.extend(base_probe.position.w),
            cascade: base_probe.cascade,
        });

        let corner_weights = std::array::from_fn::<_, 8, _>(|i| {
            let id = (i & 1) + ((i >> 1_u32) & 1) + ((i >> 2_u32) & 1);
            0.25_f32.powi(id as i32) * 0.75_f32.powi((3 - id) as i32)
        });

        (next_probe, corner_weights.expr().read(i ^ j))
    }
    #[tracked]
    pub fn next_nearest(&self) -> Expr<Probe> {
        self.shift_up(1.expr())
    }
    // All values that have `next_nearest` equal to this.
    #[tracked]
    pub fn last_nearest(&self, i: Expr<u32>) -> Expr<Probe> {
        let offset = index_to_corner(i);
        let base = self.position.xyz() << 1;
        let pos = base + offset;
        Probe::from_comps_expr(ProbeComps {
            position: pos.extend(self.position.w),
            cascade: self.cascade - 1,
        })
    }
    #[tracked]
    pub fn shift_up(&self, delta: Expr<u32>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps {
            position: (self.position.xyz() >> delta).extend(self.position.w),
            cascade: self.cascade + delta,
        })
    }
    // gets the corner probe
    #[tracked]
    pub fn shift_down(&self, delta: Expr<u32>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps {
            position: (self.position.xyz() << delta).extend(self.position.w),
            cascade: self.cascade - delta,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, Default)]
pub struct Projection {
    view_pos: Vec3<f32>,
    // Added to the LOD index; raised when the probe table is close to full.
    lod_bias: f32,
}

#[derive(Debug)]
pub struct HashCache {
    pub base_grid_size: f32,
    pub base_radius: f32,
    pub blending_radius: f32,
    pub base_angular_resolution: u32,
    pub base_probes: u32,
    pub num_cascades: u32,
    pub projection: Singleton<Projection>,
    pub table: HashTable,
    pub weighted_fluence: Buffer<Vec4<f32>>,
    pub weights: Buffer<f32>,
}
impl HashCache {
    pub fn new(
        base_grid_size: f32,
        base_radius: f32,
        blending_radius: f32,
        base_angular_resolution: u32,
        base_probes: u32,
        num_cascades: u32,
    ) -> Self {
        assert!(base_probes.is_power_of_two());
        let num_data = (base_angular_resolution.pow(2)
            * INDEX_SCALING.x
            * INDEX_SCALING.y
            * base_probes
            * num_cascades) as usize;
        Self {
            base_grid_size,
            base_radius,
            blending_radius,
            base_angular_resolution,
            base_probes,
            num_cascades,
            projection: Singleton::new(Projection::default()),

            table: HashTable::new(
                num_cascades,
                base_probes * 4, // Can use an overly large amount to reduce collisions.
                (0..num_cascades).map(|level| base_probes >> (level * 2)),
            ),
            weighted_fluence: DEVICE.create_buffer_from_fn(num_data, |_| Vec4::splat(0.0)),
            weights: DEVICE.create_buffer_from_fn(num_data, |_| 0.0),
        }
    }
    pub fn set_view(&self, view: View, lod_bias: f32) -> NodeConfigs<'static> {
        self.projection.write_host(Projection {
            view_pos: view.pos,
            lod_bias,
        })
    }
    pub fn total_probes(&self) -> u32 {
        self.table.data_capacity()
    }
    pub fn max_probes_at_cascade(&self, cascade: u32) -> u32 {
        self.base_probes >> (cascade * 2)
    }
    pub fn directions_per_probe(&self, cascade: u32) -> u32 {
        self.base_angular_resolution.pow(2)
            * INDEX_SCALING.x
            * INDEX_SCALING.y
            * (1 << (cascade * 2))
    }
    pub fn total_directions_per_cascade(&self) -> u32 {
        self.base_probes * self.base_angular_resolution.pow(2) * INDEX_SCALING.x * INDEX_SCALING.y
    }
    pub fn total_directions(&self) -> u32 {
        self.total_directions_per_cascade() * self.num_cascades
    }
    pub fn projection(&self) -> Expr<Projection> {
        self.projection.read()
    }
    #[tracked]
    pub fn get_encoder(&self, cascade: Expr<u32>) -> Expr<DirectionIndexer> {
        DirectionIndexer::from_comps_expr(DirectionIndexerComps {
            resolution: self.base_angular_resolution << cascade,
        })
    }

    pub fn get_index_offset_host(&self, cascade: u32) -> u32 {
        // base_nodes + base_nodes / 4 + base_nodes / 16 + ... (level times)
        let pow4s = (0..16).map(|i| 1_u32 << (2 * i)).sum::<u32>();
        (pow4s & ((1 << (2 * cascade)) - 1))
            << (self.base_probes.trailing_zeros() - cascade * 2 + 2)
    }

    // Probably can precompute.
    #[tracked]
    pub fn get_index_offset(&self, cascade: Expr<u32>) -> Expr<u32> {
        // base_nodes + base_nodes / 4 + base_nodes / 16 + ... (level times)
        let pow4s = (0..16).map(|i| 1_u32 << (2 * i)).sum::<u32>();
        (pow4s & ((1 << (2 * cascade)) - 1))
            << (self.base_probes.trailing_zeros() - cascade * 2 + 2)
    }
    #[tracked]
    pub fn estimate_lod(&self, pos: Expr<Vec3<f32>>) -> Expr<u32> {
        let projection = self.projection();
        let radius = (pos - projection.view_pos).abs().reduce_max();
        keter::max((radius / self.base_radius).log2() + projection.lod_bias, 0.0)
            .ceil()
            .cast_u32()
    }
    #[tracked]
    pub fn estimate_lod_secondary(&self, pos: Expr<Vec3<f32>>) -> Expr<u32> {
        self.estimate_lod(pos) + 2
    }
    #[tracked]
    pub fn iter_lods(&self, pos: Expr<Vec3<f32>>, f: impl Fn(Expr<u32>, Expr<u32>, Expr<f32>)) {
        if self.blending_radius == 1.0 {
            // Don't do LODs
            f(self.estimate_lod(pos), 0.expr(), 1.0.expr());
        } else {
            let projection = self.projection();
            let radius = (pos - projection.view_pos).abs().reduce_max();
            let lod_index = keter::max(
                (radius / self.base_radius).log2() + projection.lod_bias,
                -0.9,
            );
            let blending_factor =
                keter::min(1.0, (1.0 - lod_index.fract()) / self.blending_radius.log2());
            let lower_lod = lod_index.ceil().cast_u32();
            // let higher_lod = (lod_index + self.blending_radius.log2()).ceil().cast_u32();
            let num_lods = if blending_factor >= 1.0 {
                1u32.expr()
            } else {
                2.expr()
            };
            for i in 0u32.expr()..num_lods {
                f(
                    lower_lod + i,
                    i,
                    if i == 0 {
                        blending_factor
                    } else {
                        1.0 - blending_factor
                    },
                );
            }
        }
    }

    #[tracked]
    pub fn deposit(&self, offset: Expr<u32>, fluence: Expr<Fluence>) {
        let fl = self.weighted_fluence.atomic_ref(offset);
        fl.fetch_add(fluence);
        self.weights.atomic_ref(offset).fetch_add(1.0);
    }
    #[tracked]
    pub fn deposit_once(&self, offset: Expr<u32>, fluence: Expr<Fluence>) {
        self.weighted_fluence.write(offset, fluence);
        self.weights.write(offset, 1.0);
    }
    #[tracked]
    pub fn deposit_unstable(&self, offset: Expr<u32>, fluence: Expr<Fluence>) {
        self.weighted_fluence
            .write(offset, self.weighted_fluence.read(offset) + fluence);
        self.weights.write(offset, self.weights.read(offset) + 1.0);
    }

    #[tracked]
    pub fn quantize(
        &self,
        pos: Expr<Vec3<f32>>,
        lod: Expr<u32>,
        cascade: Expr<u32>,
    ) -> Expr<Probe> {
        let spacing = self.spacing(lod, 0);

        let pos = (pos / spacing).floor().cast_i32();
        Probe::from_comps_expr(ProbeComps {
            position: (pos.cast_u32() + OFFSET).extend(lod),
            cascade: 0.expr(),
        })
        .shift_up(cascade)
    }
    #[tracked]
    pub fn quantize_trilinear(
        &self,
        pos: Expr<Vec3<f32>>,
        lod: Expr<u32>,
        cascade: Expr<u32>,
    ) -> Expr<Trilinear> {
        let spacing = self.spacing(lod, cascade);
        let base = self.quantize(pos - spacing / 2.0, lod, cascade);
        let fract = (pos / spacing - 0.5).fract();
        Trilinear::from_comps_expr(TrilinearComps {
            base: base.position.xyz(),
            fract,
        })
    }
    #[tracked]
    pub fn dequantize(&self, probe: Expr<Probe>) -> Expr<Vec3<f32>> {
        let base_probe = probe.shift_down(probe.cascade);
        let local_pos = (base_probe.position.xyz() - OFFSET).cast_i32().cast_f32();
        let spacing = self.spacing(probe.position.w, 0);
        let offset = self.probe_spacing(probe) / 2.0;
        local_pos * spacing + offset
    }
    #[tracked]
    pub fn probe_spacing(&self, probe: Expr<Probe>) -> Expr<f32> {
        self.spacing(probe.position.w, probe.cascade)
    }
    #[tracked]
    pub fn spacing(&self, lod: Expr<u32>, cascade: impl AsExpr<Value = u32>) -> Expr<f32> {
        self.base_grid_size * (1_u32 << (cascade.as_expr() + lod)).cast_f32()
    }
    #[tracked]
    pub fn decode(&self, encoded_pos: Expr<Key>, cascade: Expr<u32>) -> Expr<Probe> {
        Probe::from_comps_expr(ProbeComps {
            position: decode_position(encoded_pos),
            cascade,
        })
    }
    #[tracked]
    pub fn encode(&self, probe: Expr<Probe>) -> Expr<Key> {
        encode_position(probe.position)
    }
    #[tracked]
    pub fn load_index(&self, probe: Expr<Probe>, index: Expr<u32>) -> Expr<LoadedNode> {
        let encoder = self.get_encoder(probe.cascade);
        let data_offset = if index == u32::MAX {
            0_u32.expr()
        } else {
            probe.cascade * self.total_directions_per_cascade()
                + (index - self.get_index_offset(probe.cascade)) * encoder.linear_size()
        };
        LoadedNode::from_comps_expr(LoadedNodeComps {
            probe,
            data_offset,
            index,
        })
    }
    #[tracked]
    pub fn insert(&self, probe: Expr<Probe>, on_insert: impl Fn(Expr<u32>)) {
        self.table
            .insert(self.encode(probe), probe.cascade, on_insert);
    }
    #[tracked]
    pub fn get(&self, probe: Expr<Probe>) -> Expr<LoadedNode> {
        let index = self.table.get(self.encode(probe), probe.cascade);
        self.load_index(probe, index)
    }
    #[tracked]
    pub fn get_occupied(&self, cascade: Expr<u32>, index: Expr<u32>) -> Expr<LoadedNode> {
        let index = self.get_index_offset(cascade) + index;
        let key = self.table.data_key.read(index);
        let probe = self.decode(key, cascade);
        if key == INVALID_KEY {
            LoadedNode::from_comps_expr(LoadedNodeComps {
                probe,
                data_offset: 0.expr(),
                index: u32::MAX.expr(),
            })
        } else {
            self.load_index(probe, index)
        }
    }

    #[tracked]
    pub fn contains(&self, probe: Expr<Probe>) -> Expr<bool> {
        self.table.contains(self.encode(probe), probe.cascade)
    }
    #[tracked]
    pub fn cost(&self, probe: Expr<Probe>) -> Expr<u32> {
        self.table.cost(self.encode(probe), probe.cascade)
    }

    #[tracked]
    pub fn sample(
        &self,
        pos: Expr<Vec3<f32>>,
        cascade: Expr<u32>,
        dir: Expr<Vec3<f32>>,
    ) -> Expr<Fluence> {
        let lod = self.estimate_lod(pos);
        let interp_probes = self.quantize_trilinear(pos, lod, cascade);

        let encoder = self.get_encoder(cascade);
        let interp_dirs = encoder.dir_to_grid_bilinear(dir);

        let fluence = Vec4::splat(0.0_f32).var();
        let total_weight = 0.0_f32.var();

        for j in 0u32..8 {
            let (probe, probe_weight) = interp_probes.get(j);
            let probe = Probe::from_comps_expr(ProbeComps {
                position: probe.extend(lod),
                cascade,
            });
            let node = self.get(probe);
            if node.loaded() {
                for i in 0u32..4 {
                    let (dir, weight) = interp_dirs.get(i);
                    let offset = node.data_offset + encoder.grid_to_linear(dir);
                    *fluence += self.weighted_fluence.read(offset) * weight * probe_weight;
                    *total_weight += self.weights.read(offset) * weight * probe_weight;
                }
            }
        }

        let output = fluence / total_weight;
        output
    }
}
