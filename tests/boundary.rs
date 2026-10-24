//! MicroAgent must stay usable on its own: MIT, one dependency, and no knowledge of the app it
//! was first written for. These tests fail if that slips, so the crate can be moved to its own
//! repository at any time.

use std::fs;
use std::path::Path;

fn crate_file(rel: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).unwrap()
}

#[test]
fn it_is_mit_licensed_and_depends_on_serde_json_only() {
    let manifest = crate_file("Cargo.toml");
    assert!(manifest.contains(r#"license = "MIT""#), "the crate must say it is MIT");
    assert!(crate_file("LICENSE").starts_with("MIT License"));
    let deps = manifest.split("[dependencies]").nth(1).unwrap_or("");
    let names: Vec<&str> = deps.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('[')).map(|l| l.split(['=', ' ']).next().unwrap()).collect();
    assert_eq!(names, vec!["serde_json"], "keep the dependency list this short");
    assert!(!manifest.contains("path ="), "no path dependencies: the crate must be movable");
}

#[test]
fn its_source_does_not_mention_the_app_it_came_from() {
    for file in ["src/lib.rs", "src/agent.rs", "src/tool.rs", "src/host.rs", "README.md"] {
        let text = crate_file(file).to_lowercase();
        for word in ["center", "lumen", "spreadsheet", "wasm_bindgen"] {
            assert!(!text.contains(word), "{file} mentions {word}: keep MicroAgent free of app details");
        }
    }
}
