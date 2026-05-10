import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from config.settings import BASE_DIR, LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN


# Approximate local landmarks for readable hotspot labels in the demo UI.
# Coordinates are only used for naming nearby grids, not for navigation.
LANDMARKS = [
    {"name": "水泊梁山风景区周边", "lng": 116.09895, "lat": 35.78690, "type": "景区"},
    {"name": "梁山县人民政府周边", "lng": 116.09620, "lat": 35.80290, "type": "政务"},
    {"name": "梁山汽车站周边", "lng": 116.09100, "lat": 35.78900, "type": "交通"},
    {"name": "水泊南路商圈周边", "lng": 116.09180, "lat": 35.79350, "type": "商圈"},
    {"name": "梁山站周边", "lng": 116.11800, "lat": 35.83200, "type": "交通"},
    {"name": "梁山县人民医院周边", "lng": 116.08350, "lat": 35.80250, "type": "医疗"},
    {"name": "梁山一中周边", "lng": 116.10500, "lat": 35.80900, "type": "学校"},
    {"name": "西环路公交场站周边", "lng": 116.06400, "lat": 35.79900, "type": "交通"},
]


def haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    radius = 6371.0
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def nearest_area_name(lng: float, lat: float) -> tuple[str, float, str]:
    best = min(
        LANDMARKS,
        key=lambda item: haversine_km(float(lng), float(lat), item["lng"], item["lat"]),
    )
    distance = haversine_km(float(lng), float(lat), best["lng"], best["lat"])
    return best["name"], round(distance, 2), best["type"]


def add_area_names(df):
    if df.empty or "center_lng" not in df.columns or "center_lat" not in df.columns:
        return df
    result = df.copy()
    names = result.apply(lambda row: nearest_area_name(row["center_lng"], row["center_lat"]), axis=1)
    result["区域名称"] = names.map(lambda item: item[0])
    result["距地标km"] = names.map(lambda item: item[1])
    result["区域类型"] = names.map(lambda item: item[2])
    return result


def liangshan_boundary_polygon() -> list[list[float]]:
    """Approximate demo boundary from configured Liangshan coordinate range."""
    return [
        [LAT_MIN, LNG_MIN],
        [LAT_MIN, LNG_MAX],
        [LAT_MAX, LNG_MAX],
        [LAT_MAX, LNG_MIN],
        [LAT_MIN, LNG_MIN],
    ]


def liangshan_geojson_path() -> Path | None:
    path = BASE_DIR / "ls.geoJson"
    return path if path.exists() else None


def liangshan_bounds() -> tuple[float, float, float, float]:
    """Return min_lng, min_lat, max_lng, max_lat."""
    path = liangshan_geojson_path()
    if not path:
        return LNG_MIN, LAT_MIN, LNG_MAX, LAT_MAX
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        coords = []

        def collect(obj):
            if isinstance(obj, list) and len(obj) >= 2 and all(isinstance(x, (int, float)) for x in obj[:2]):
                coords.append((float(obj[0]), float(obj[1])))
            elif isinstance(obj, list):
                for item in obj:
                    collect(item)

        for feature in data.get("features", []):
            collect(feature.get("geometry", {}).get("coordinates", []))
        if coords:
            lngs = [item[0] for item in coords]
            lats = [item[1] for item in coords]
            return min(lngs), min(lats), max(lngs), max(lats)
    except Exception:
        pass
    return LNG_MIN, LAT_MIN, LNG_MAX, LAT_MAX
