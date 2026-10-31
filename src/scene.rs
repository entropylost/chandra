use std::path::Path;

use serde::Deserialize;

use crate::algorithm::WorldData;
use crate::algorithm::rc::RcConfig;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub pos: glam::Vec3,
}

#[derive(Debug, Deserialize)]
pub struct Scene {
    pub world: WorldData,
    #[serde(default)]
    pub rc_config: RcConfig,
    pub ddgi_bounds: (glam::Vec3, glam::Vec3),
    pub ddgi_probe_dims: glam::UVec3,
    pub camera: Camera,
}
impl Scene {
    pub fn load_file(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("Failed to open scene {}: {e}", path.display()));
        serde_json::from_reader(std::io::BufReader::new(file))
            .unwrap_or_else(|e| panic!("Failed to parse scene {}: {e}", path.display()))
    }
}
