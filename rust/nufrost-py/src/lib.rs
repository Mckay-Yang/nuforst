use pyo3::prelude::*;

/// Adds two integers (Python-callable).
#[pyfunction]
fn py_add(left: i32, right: i32) -> PyResult<i32> {
    Ok(nufrost_core::add(left, right))
}

/// A Python module implemented in Rust.
#[pymodule]
fn nufrost_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_add, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_no_py() {
        assert_eq!(2_usize + 2, 4);
    }
}
