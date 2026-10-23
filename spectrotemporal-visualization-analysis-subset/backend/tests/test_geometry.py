from stv_core.geometry import buffered_bbox_polygon, compute_bbox, convex_hull, point_in_polygon


def test_convex_hull_wraps_outer_points():
    points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
    hull = convex_hull(points)
    assert hull == [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]


def test_compute_bbox_returns_extents():
    bbox = compute_bbox([(138.68, -34.53), (138.69, -34.52), (138.67, -34.54)])
    assert bbox == {
        "min_lon": 138.67,
        "max_lon": 138.69,
        "min_lat": -34.54,
        "max_lat": -34.52,
    }


def test_buffered_bbox_polygon_contains_points_and_buffer():
    polygon = buffered_bbox_polygon([(138.689, -34.525), (138.6892, -34.5251)], 10)

    assert len(polygon) == 4
    assert point_in_polygon((138.689, -34.525), polygon)
    assert point_in_polygon((138.6892, -34.5251), polygon)
    assert not point_in_polygon((138.6905, -34.524), polygon)
