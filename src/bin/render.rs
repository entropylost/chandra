use std::path::PathBuf;

use chandra::scene::Scene;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Renders a scene using Split RC or DDGI")]
struct Args {
    #[arg(short, long, default_value_t = 1)]
    frames: u32,
    #[arg(short, long, default_value_t = 1)]
    grid_size_mult: u32,
    #[arg(short, long)]
    ddgi: bool,
    #[arg(short, long)]
    multibounce: bool,
    scene: String,
}

fn main() {
    let args = Args::parse();
    let path = if args.scene.ends_with(".json") {
        PathBuf::from(&args.scene)
    } else {
        PathBuf::from(format!("scenes/{}.json", args.scene))
    };
    let scene = Scene::load_file(path);
    
}
