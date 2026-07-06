use std::collections::BTreeSet;
use std::path::PathBuf;

use gdal::full_scene::{discover_sentinel_band_stacks, sorted_band_names};
use gdal::scene_cache::{build_scene_cache, load_scene_cache, load_scene_from_geotiffs_window};
use gdal::{extract_timestamps_from_band_descriptions, read_all_bands, RasterReader};

const SOURCE_NAME: &str = "sentinel-2";
const EXPECTED_TIMESTEPS: usize = 1076;
const EXPECTED_BANDS: &[&str] = &["B2", "B3", "B4", "B8", "B11", "B12"];

#[derive(Debug, Clone, Copy)]
struct FixtureScene {
    lon: f64,
    lat: f64,
    size: usize,
}

const FIXTURES: &[FixtureScene] = &[
    FixtureScene {
        lon: 100.1121,
        lat: 25.6541,
        size: 1,
    },
    FixtureScene {
        lon: 100.1122,
        lat: 25.6542,
        size: 2,
    },
    FixtureScene {
        lon: 100.1123,
        lat: 25.6543,
        size: 3,
    },
];

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

fn raw_fixture_root() -> PathBuf {
    test_data_root()
        .join("raw")
        .join(SOURCE_NAME)
        .join("16-sites")
}

fn test_output_root(name: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join("test-output")
        .join(name)
}

fn expected_band_strings() -> Vec<String> {
    EXPECTED_BANDS
        .iter()
        .map(|band| (*band).to_string())
        .collect()
}

#[test]
fn sentinel_time_series_fixtures_are_complete_and_readable() {
    let raw_root = raw_fixture_root();
    assert!(
        raw_root.is_dir(),
        "fixture raw root should exist: {}",
        raw_root.display()
    );

    for fixture in FIXTURES {
        let stacks = discover_sentinel_band_stacks(&raw_root, fixture.lon, fixture.lat)
            .expect("fixture discovery should succeed");
        assert_eq!(
            sorted_band_names(&stacks),
            EXPECTED_BANDS,
            "fixture {fixture:?} should expose the expected band order"
        );

        for band in EXPECTED_BANDS {
            let paths = stacks.get(*band).expect("band should exist");
            assert_eq!(paths.len(), 1, "fixture {fixture:?} band {band}");

            let reader = RasterReader::open(&paths[0]).expect("fixture tif should open");
            assert_eq!(reader.shape(), (fixture.size, fixture.size));
            assert_eq!(reader.band_count(), EXPECTED_TIMESTEPS);

            let (days, target) = extract_timestamps_from_band_descriptions(&reader)
                .expect("fixture band descriptions should contain timestamps");
            assert_eq!(days.len(), EXPECTED_TIMESTEPS);
            assert_eq!(
                target,
                *days.last().expect("timestamp axis should not be empty")
            );

            let first = reader.read_band(1).expect("first timestep should read");
            assert_eq!(first.dim(), (fixture.size, fixture.size));
            let cube = read_all_bands(&reader).expect("fixture time series should read");
            assert_eq!(cube.dim(), (EXPECTED_TIMESTEPS, fixture.size, fixture.size));
            assert!(
                cube.iter().any(|value| value.is_finite()),
                "fixture {fixture:?} band {band} should contain finite data over its time series"
            );
        }
    }
}

#[test]
fn scene_cache_builds_from_new_test_data_layout() {
    let fixture = FIXTURES
        .iter()
        .find(|fixture| fixture.size == 2)
        .expect("2x2 fixture should exist");
    let cache_root = test_output_root("gdal-scene-cache")
        .join("cache")
        .join(SOURCE_NAME)
        .join("16-sites");
    let _ = std::fs::remove_dir_all(test_output_root("gdal-scene-cache"));

    let cache_dir = build_scene_cache(
        &test_data_root(),
        &cache_root,
        SOURCE_NAME,
        fixture.lon,
        fixture.lat,
        None,
    )
    .expect("scene cache should build from tests/data/raw");

    assert!(cache_dir.join("meta.json").is_file());
    assert!(cache_dir.join("cube.f32.bin").is_file());

    let loaded = load_scene_cache(&cache_dir).expect("fresh scene cache should load");
    assert_eq!(loaded.ordered_bands, expected_band_strings());
    assert_eq!(loaded.band_cubes.len(), EXPECTED_BANDS.len());

    for band in EXPECTED_BANDS {
        let cube = loaded
            .band_cubes
            .get(*band)
            .expect("band cube should exist");
        assert_eq!(cube.dim(), (EXPECTED_TIMESTEPS, fixture.size, fixture.size));
        let timestamps = loaded
            .band_timestamps
            .get(*band)
            .expect("band timestamps should exist");
        assert_eq!(timestamps.len(), EXPECTED_TIMESTEPS);
    }
}

#[test]
fn geotiff_window_loader_reads_subset_from_fixture_scene() {
    let fixture = FIXTURES
        .iter()
        .find(|fixture| fixture.size == 3)
        .expect("3x3 fixture should exist");

    let loaded = load_scene_from_geotiffs_window(
        &test_data_root(),
        SOURCE_NAME,
        fixture.lon,
        fixture.lat,
        2,
        None,
        None,
    )
    .expect("windowed fixture load should succeed");

    assert_eq!(loaded.ordered_bands, expected_band_strings());
    assert!(loaded.cache_dir.is_none());

    let observed: BTreeSet<&str> = loaded.band_cubes.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = EXPECTED_BANDS.iter().copied().collect();
    assert_eq!(observed, expected);

    for band in EXPECTED_BANDS {
        let cube = loaded
            .band_cubes
            .get(*band)
            .expect("band cube should exist");
        assert_eq!(cube.dim(), (EXPECTED_TIMESTEPS, 2, 2));
        assert!(
            cube.iter().any(|value| value.is_finite()),
            "windowed cube for {band} should contain finite data"
        );
    }
}
