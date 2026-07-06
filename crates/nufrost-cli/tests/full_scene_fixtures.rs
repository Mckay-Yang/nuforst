use std::path::PathBuf;
use std::process::Command;

const SOURCE_NAME: &str = "sentinel-2";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has workspace parent")
        .parent()
        .expect("crates dir has workspace parent")
        .to_path_buf()
}

fn test_data_root() -> PathBuf {
    workspace_root().join("tests/data")
}

fn test_output_root(name: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join("test-output")
        .join(name)
}

#[test]
fn full_scene_cli_runs_against_1x1_time_series_fixture() {
    let output_root = test_output_root("nufrost-cli-full-scene-1x1");
    let cache_dir = test_data_root()
        .join("cache")
        .join(SOURCE_NAME)
        .join("16-sites")
        .join("lon100.112100_lat25.654100");
    let _ = std::fs::remove_dir_all(&output_root);
    let _ = std::fs::remove_dir_all(&cache_dir);

    let status = Command::new(env!("CARGO_BIN_EXE_nufrost-cli"))
        .args([
            "full-scene",
            "--source-name",
            SOURCE_NAME,
            "--lon",
            "100.1121",
            "--lat",
            "25.6541",
            "--data-root",
        ])
        .arg(test_data_root())
        .arg("--output-root")
        .arg(&output_root)
        .args([
            "--methods",
            "nufrost",
            "--n-jobs",
            "1",
            "--min-valid-ratio",
            "0.5",
        ])
        .status()
        .expect("nufrost-cli should launch");

    assert!(status.success(), "nufrost-cli exited with {status}");
    assert!(cache_dir.join("meta.json").is_file());
    assert!(cache_dir.join("cube.f32.bin").is_file());

    let scene_dir = output_root
        .join(format!("{SOURCE_NAME}_recon"))
        .join("100.1121_25.6541");
    assert!(scene_dir.is_dir(), "scene output dir should exist");

    let mut predictions = Vec::new();
    let mut ground_truths = Vec::new();
    for entry in std::fs::read_dir(&scene_dir).expect("scene dir should be readable") {
        let path = entry.expect("scene output entry should read").path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with("[nufrost]_") && name.ends_with("_prediction.tif") {
            predictions.push(path);
        } else if name.starts_with("[ground_truth]_") && name.ends_with(".tif") {
            ground_truths.push(path);
        }
    }
    assert_eq!(
        predictions.len(),
        1,
        "one NUFROST prediction should be written"
    );
    assert_eq!(
        ground_truths.len(),
        1,
        "one ground-truth stack should be written"
    );

    let summary_dir = output_root.join("run_summaries");
    let summaries: Vec<_> = std::fs::read_dir(&summary_dir)
        .expect("summary dir should be readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("reconstruction_summary_sentinel-2_"))
        })
        .collect();
    assert_eq!(
        summaries.len(),
        1,
        "one reconstruction summary should be written"
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&summaries[0]).expect("summary should read"))
            .expect("summary JSON should parse");
    assert_eq!(summary["source_name"], SOURCE_NAME);
    assert!(summary["metrics"]["nufrost"]["overall_rmse"]
        .as_f64()
        .is_some_and(f64::is_finite));
}
