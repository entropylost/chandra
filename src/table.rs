use keter::lang::types::vector::Vec3;
use keter::prelude::Value as _; // Shadowed for some reason
use keter::prelude::*;
use yesod::device_println;

pub type Key = u64;

pub const INVALID_KEY: Key = Key::MAX;

#[tracked]
fn hash(key: Expr<Key>) -> Expr<u32> {
    yesod::rand::pcg3d(Vec3::expr((key >> 32).cast_u32(), key.cast_u32(), 937)).x
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
#[repr(C)]
pub struct Value {
    index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
#[repr(C)]
pub struct Entry {
    key: Key,
    value: Value,
}
impl Entry {
    pub const EMPTY: Self = Self {
        key: INVALID_KEY,
        value: Value { index: u32::MAX },
    };
}

#[derive(Debug)]
pub struct HashTable {
    pub num_tables: u32,
    pub entry_capacity: u32,
    pub data_capacity: u32,
    // per entry
    pub entries: Buffer<Entry>,
    // per datum
    pub freelist: Buffer<u32>,
    pub data_key: Buffer<Key>,
    // per table
    pub table_data_start: Buffer<u32>,
    pub table_data_end: Buffer<u32>,
    pub freelist_index: Buffer<u32>, // reset to table_data_start after each build
    pub last_occupied_index: Buffer<u32>,
    pub total_occupied: Buffer<u32>,
}

/*
update:
- reset freelist_index to table_data_start
- clear hashmap entries
- call build_freelists (repopulates hashmap and freelist)
- reset freelist_index to table_data_start (also want to check the max just in case?)
- perform insertions
- perform queries
*/
impl HashTable {
    pub fn new(
        num_tables: u32,
        entry_capacity: u32,
        data_capacities: impl Iterator<Item = u32>,
    ) -> Self {
        let total_entry_capacity = entry_capacity * num_tables;
        let mut table_data_start = Vec::new();
        let mut table_data_end = Vec::new();
        let mut total_data_capacity = 0;
        for cap in data_capacities {
            table_data_start.push(total_data_capacity);
            total_data_capacity += cap;
            table_data_end.push(total_data_capacity);
        }
        assert_eq!(table_data_start.len() as u32, num_tables);
        Self {
            num_tables,
            entry_capacity,
            data_capacity: total_data_capacity,

            entries: DEVICE.create_buffer_from_fn(total_entry_capacity as usize, |_| Entry::EMPTY),
            freelist: DEVICE.create_buffer(total_data_capacity as usize),
            data_key: DEVICE.create_buffer_from_fn(total_data_capacity as usize, |_| INVALID_KEY),

            table_data_start: DEVICE.create_buffer_from_slice(&table_data_start),
            table_data_end: DEVICE.create_buffer_from_slice(&table_data_end),
            freelist_index: DEVICE.create_buffer(num_tables as usize),
            last_occupied_index: DEVICE.create_buffer(num_tables as usize),
            total_occupied: DEVICE.create_buffer(num_tables as usize),
        }
    }
    pub fn memory_size(&self) -> usize {
        self.entries.size_bytes()
            + self.freelist.size_bytes()
            + self.data_key.size_bytes()
            + self.table_data_start.size_bytes()
            + self.freelist_index.size_bytes()
            + self.last_occupied_index.size_bytes()
    }
    pub fn data_capacity(&self) -> u32 {
        self.data_capacity
    }
    pub fn entry_capacity(&self) -> u32 {
        self.entry_capacity
    }
    fn probing_step(&self) -> u32 {
        1 // (GOLDEN_RATIO * self.entry_capacity as f32) as u32
    }
    #[tracked]
    fn insert_entry_index(&self, key: Expr<Key>, table: Expr<u32>, index: Expr<u32>) {
        let entry_index = (hash(key) % self.entry_capacity).var();
        let offset = self.entry_capacity * table;
        loop {
            let entry = self.entries.atomic_ref(entry_index + offset);
            let prev = entry.key.compare_exchange(INVALID_KEY, key);
            if prev == key {
                // already inserted, do nothing
                device_println!("Error: duplicate key insertion should be impossible");
                break;
            } else if prev == INVALID_KEY {
                entry.value.index.exchange(index);
                break;
            }

            *entry_index = (entry_index + self.probing_step()) % self.entry_capacity;
        }
    }
    // Inserts a key into the table. It takes a while to get the index, so that has to be done later after synchronization.
    #[tracked]
    pub fn insert(&self, key: Expr<Key>, table: Expr<u32>, on_insert: impl Fn(Expr<u32>)) {
        let entry_index = (hash(key) % self.entry_capacity).var();
        let offset = self.entry_capacity * table;
        loop {
            let entry = self.entries.atomic_ref(offset + entry_index);
            let prev = entry.key.compare_exchange(INVALID_KEY, key);
            if prev == key {
                // already inserted, do nothing
                break;
            } else if prev == INVALID_KEY {
                let freelist_index = self.freelist_index.atomic_ref(table).fetch_add(1);
                // Free slots are [start, end - occupied); past that the table is full, so
                // leave the entry unloaded (index u32::MAX) rather than stealing another table's slots.
                let free_end =
                    self.table_data_end.read(table) - self.total_occupied.read(table);
                if freelist_index >= free_end {
                    break;
                }
                let index = self.freelist.read(freelist_index);
                entry.value.index.exchange(index);
                self.data_key.write(index, key);
                on_insert(index);
                break;
            }
            *entry_index = (entry_index + self.probing_step()) % self.entry_capacity;
        }
    }
    #[tracked]
    pub fn get(&self, key: Expr<Key>, table: Expr<u32>) -> Expr<u32> {
        let entry_index = (hash(key) % self.entry_capacity).var();
        let offset = self.entry_capacity * table;
        let result = u32::MAX.var();
        loop {
            let entry = self.entries.read(offset + entry_index);
            if entry.key == key {
                *result = entry.value.index;
                break;
            } else if entry.key == INVALID_KEY {
                break;
            }
            *entry_index = (entry_index + self.probing_step()) % self.entry_capacity;
        }
        **result
    }
    #[tracked]
    pub fn contains(&self, key: Expr<Key>, table: Expr<u32>) -> Expr<bool> {
        self.get(key, table) != u32::MAX
    }
    #[tracked]
    pub fn cost(&self, key: Expr<Key>, table: Expr<u32>) -> Expr<u32> {
        let entry_index = (hash(key) % self.entry_capacity).var();
        let offset = self.entry_capacity * table;
        let cost = 0_u32.var();
        loop {
            let entry = self.entries.read(offset + entry_index);
            if entry.key == key {
                break;
            } else if entry.key == INVALID_KEY {
                *cost = u32::MAX;
                break;
            }
            *entry_index = (entry_index + self.probing_step()) % self.entry_capacity;
            *cost += 1;
        }
        **cost
    }
    // remove: if true, then item is removed from the table
    #[tracked]
    pub fn build_freelists(
        &self,
        data_index: Expr<u32>,
        remove: impl Fn(Expr<Key>, Expr<u32>) -> Expr<bool>,
    ) {
        let table = 0_u32.var();
        // TODO: Can be optimized if we know more about the table sizes
        for i in 1..self.num_tables {
            let t = self.table_data_start.read(i);
            if data_index >= t {
                *table = i;
            } else {
                break;
            }
        }
        let key = self.data_key.read(data_index);
        let occupied = (key != INVALID_KEY).var();
        // Separate?
        if occupied {
            if remove(key, **table) {
                self.data_key.write(data_index, INVALID_KEY);
                *occupied = false;
            } else {
                let key = self.data_key.read(data_index);
                self.insert_entry_index(key, **table, data_index);
                self.last_occupied_index
                    .atomic_ref(table)
                    .fetch_max(data_index);
                self.total_occupied.atomic_ref(table).fetch_add(1);
            }
        }
        if !occupied {
            let freelist_index = self.freelist_index.atomic_ref(table).fetch_add(1);
            self.freelist.write(freelist_index, data_index);
        }
    }
    // #[tracked]
    // pub fn get_occupied_index(&self, table: Expr<u32>, index: Expr<u32>) -> Expr<u32> {
    //     self.freelist
    //         .read(self.table_data_start.read(table) + index)
    // }
}
