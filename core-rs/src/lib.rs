use pyo3::prelude::*;

mod geometry;

use geometry::Points;

#[pyfunction]
fn polygon_area(points: Points) -> f64 {
    geometry::area(&points)
}

#[pyfunction]
fn occupancy(py: Python<'_>, spots: Vec<Points>, vehicles: Vec<Points>) -> Vec<f64> {
    // pure geometry, no Python objects: release the GIL
    py.allow_threads(|| geometry::occupancy(&spots, &vehicles))
}

#[pymodule]
fn parking_logic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(polygon_area, m)?)?;
    m.add_function(wrap_pyfunction!(occupancy, m)?)?;
    Ok(())
}
