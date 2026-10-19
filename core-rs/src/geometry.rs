use geo::{Area, BooleanOps, Coord, LineString, MultiPolygon, Polygon};

pub type Points = Vec<(f64, f64)>;

pub fn to_polygon(points: &Points) -> Polygon<f64> {
    let coords: Vec<Coord<f64>> = points.iter().map(|&(x, y)| Coord { x, y }).collect();
    Polygon::new(LineString::from(coords), vec![])
}

pub fn area(points: &Points) -> f64 {
    to_polygon(points).unsigned_area()
}

pub fn union(polygons: &[Points]) -> MultiPolygon<f64> {
    polygons
        .iter()
        .map(|p| MultiPolygon::new(vec![to_polygon(p)]))
        .fold(MultiPolygon::new(vec![]), |acc, p| acc.union(&p))
}

/// O_i = A(spot intersect union(vehicles)) / A(spot)
pub fn occupancy(spots: &[Points], vehicles: &[Points]) -> Vec<f64> {
    let vehicles = union(vehicles);
    spots
        .iter()
        .map(|s| {
            let spot = to_polygon(s);
            let spot_area = spot.unsigned_area();
            if spot_area == 0.0 {
                return 0.0;
            }
            let covered = MultiPolygon::new(vec![spot]).intersection(&vehicles);
            covered.unsigned_area() / spot_area
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Points {
        vec![(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    }

    #[test]
    fn area_of_square() {
        assert!((area(&square(0.0, 0.0, 2.0)) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn half_covered_spot() {
        let spots = vec![square(0.0, 0.0, 2.0)];
        let vehicles = vec![square(1.0, 0.0, 2.0)];
        assert!((occupancy(&spots, &vehicles)[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn overlapping_vehicles_not_counted_twice() {
        let spots = vec![square(0.0, 0.0, 2.0)];
        let vehicles = vec![square(0.0, 0.0, 2.0), square(0.0, 0.0, 2.0)];
        assert!((occupancy(&spots, &vehicles)[0] - 1.0).abs() < 1e-9);
    }
}
