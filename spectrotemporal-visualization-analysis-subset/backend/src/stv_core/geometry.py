from __future__ import annotations

from math import cos, pi


def _meters_per_degree_latitude() -> float:
    return 111_320.0


def _meters_per_degree_longitude(latitude: float) -> float:
    return max(1e-9, 111_320.0 * cos(latitude * pi / 180.0))


def compute_bbox(points: list[tuple[float, float]]) -> dict[str, float]:
    if not points:
        raise ValueError("cannot compute bbox for empty point set")
    lons = [lon for lon, _ in points]
    lats = [lat for _, lat in points]
    return {
        "min_lon": min(lons),
        "max_lon": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats),
    }


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def hull_geojson(points: list[tuple[float, float]]) -> dict[str, object]:
    hull = convex_hull(points)
    if len(hull) < 3:
        hull = buffered_bbox_polygon(points, 1.0)
    closed = hull + [hull[0]]
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[list(point) for point in closed]],
        },
    }


def buffered_bbox_polygon(points: list[tuple[float, float]], buffer_meters: float) -> list[tuple[float, float]]:
    if not points:
        raise ValueError("cannot build buffered bbox polygon for empty point set")

    lons = [lon for lon, _ in points]
    lats = [lat for _, lat in points]
    latitude = sum(lats) / len(lats)
    lon_padding = buffer_meters / _meters_per_degree_longitude(latitude)
    lat_padding = buffer_meters / _meters_per_degree_latitude()

    min_lon = min(lons) - lon_padding
    max_lon = max(lons) + lon_padding
    min_lat = min(lats) - lat_padding
    max_lat = max(lats) + lat_padding

    return [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
    ]


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    for index, left in enumerate(polygon):
        right = polygon[(index + 1) % len(polygon)]
        x1, y1 = left
        x2, y2 = right

        crosses = (y1 > y) != (y2 > y)
        if not crosses:
            continue

        x_intersection = ((x2 - x1) * (y - y1) / (y2 - y1)) + x1
        if x < x_intersection:
            inside = not inside
    return inside
