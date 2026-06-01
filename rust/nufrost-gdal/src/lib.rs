// nufrost-gdal — raster I/O via the GDAL crate.
// Requires libgdal system library (e.g. `brew install gdal` or conda).
// This is a placeholder skeleton.

/// Placeholder: returns the GDAL version string at runtime.
pub fn gdal_version() -> String {
    gdal::version::version_info("GDAL_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_gdal_version() {
        let ver = gdal_version();
        assert!(ver.contains("."), "Expected dotted version, got: {ver}");
    }
}
