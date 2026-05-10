import json

import folium
import pandas as pd
from folium.plugins import HeatMap

from config.settings import LIANGSHAN_CENTER
from modules.geo_context import liangshan_boundary_polygon, liangshan_bounds, liangshan_geojson_path, nearest_area_name


def base_map(zoom_start: int = 13) -> folium.Map:
    m = folium.Map(location=[LIANGSHAN_CENTER[1], LIANGSHAN_CENTER[0]], zoom_start=zoom_start, tiles="OpenStreetMap")
    add_liangshan_boundary(m)
    return m


def add_map_legend(m: folium.Map, title: str, items: list[tuple[str, str]]) -> None:
    rows = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
          <span style="width:12px;height:12px;border-radius:50%;background:{color};display:inline-block;"></span>
          <span>{label}</span>
        </div>
        """
        for color, label in items
    )
    legend = f"""
    <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999;
                background: rgba(255,255,255,.94); border:1px solid #cbd5e1;
                border-radius:8px; padding:10px 12px; font-size:13px;
                box-shadow:0 8px 18px rgba(15,23,42,.16);">
      <b>{title}</b>
      {rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))


def add_liangshan_boundary(m: folium.Map) -> None:
    geojson_path = liangshan_geojson_path()
    if geojson_path:
        geojson_data = json.loads(geojson_path.read_text(encoding="utf-8-sig"))
        folium.GeoJson(
            geojson_data,
            name="梁山县行政边界",
            style_function=lambda _: {
                "color": "#d7191c",
                "weight": 4,
                "fillColor": "#fee08b",
                "fillOpacity": 0.08,
            },
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["区域"]),
        ).add_to(m)
    else:
        folium.Polygon(
            locations=liangshan_boundary_polygon(),
            color="#d7191c",
            weight=4,
            fill=True,
            fill_color="#fee08b",
            fill_opacity=0.08,
            tooltip="梁山县研究区域边界（演示范围）",
        ).add_to(m)


def order_heatmap(df: pd.DataFrame, point_type: str = "end") -> folium.Map:
    m = base_map()
    if df.empty:
        return m
    lng_col, lat_col = (f"{point_type}_lng", f"{point_type}_lat")
    valid = df[[lat_col, lng_col]].dropna()
    points = valid.values.tolist()
    if points:
        m.location = [float(valid[lat_col].mean()), float(valid[lng_col].mean())]
        m.fit_bounds([[float(valid[lat_col].min()), float(valid[lng_col].min())], [float(valid[lat_col].max()), float(valid[lng_col].max())]])
        HeatMap(points, radius=13, blur=20, min_opacity=0.38, max_zoom=16).add_to(m)
        color = "#2563eb" if point_type == "start" else "#dc2626"
        sample = valid.head(800)
        for row in sample.itertuples(index=False):
            folium.CircleMarker(
                location=[float(getattr(row, lat_col)), float(getattr(row, lng_col))],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.45,
                opacity=0.35,
                weight=1,
            ).add_to(m)
        add_map_legend(
            m,
            "订单热力图说明",
            [
                ("#2563eb" if point_type == "start" else "#dc2626", "采样订单点位"),
                ("#f97316", "颜色越集中表示订单越密集"),
            ],
        )
    return m


def hotspot_map(hotspots: pd.DataFrame) -> folium.Map:
    m = base_map()
    if hotspots.empty:
        return m
    colors = {"高": "red", "中": "orange", "低": "green"}
    for row in hotspots.itertuples():
        area_name, landmark_distance, area_type = nearest_area_name(row.center_lng, row.center_lat)
        folium.Marker(
            location=[row.center_lat, row.center_lng],
            icon=folium.Icon(color=colors.get(row.hotspot_level, "blue"), icon="bicycle", prefix="fa"),
            tooltip=f"{area_name}：{row.hotspot_level}热点",
            popup=(
                f"区域: {area_name}<br>类型: {area_type}<br>距参考地标: {landmark_distance} km<br>"
                f"网格: {row.grid_id}<br>时间: {row.stat_time}<br>结束订单数量: {row.end_count}<br>"
                f"等级: {row.hotspot_level}<br>得分: {row.hotspot_score}"
            ),
        ).add_to(m)
    add_map_legend(m, "停放热点等级", [("#ef4444", "高热点"), ("#f59e0b", "中热点"), ("#22c55e", "低热点")])
    return m


def prediction_map(predictions: pd.DataFrame) -> folium.Map:
    m = base_map()
    if predictions.empty:
        return m
    colors = {"高": "red", "中": "orange", "低": "green"}
    for row in predictions.itertuples():
        area_name, landmark_distance, area_type = nearest_area_name(row.center_lng, row.center_lat)
        folium.Marker(
            location=[row.center_lat, row.center_lng],
            icon=folium.Icon(color=colors.get(row.risk_level, "blue"), icon="bicycle", prefix="fa"),
            tooltip=f"{area_name}：{row.risk_level}风险",
            popup=(
                f"区域: {area_name}<br>类型: {area_type}<br>距参考地标: {landmark_distance} km<br>"
                f"网格: {row.grid_id}<br>预测时间: {row.predict_time}<br>"
                f"预测结束订单数量: {row.predicted_end_count}<br>预测开始订单数量: {row.predicted_start_count}<br>"
                f"风险: {row.risk_level}"
            ),
        ).add_to(m)
    add_map_legend(m, "预测风险等级", [("#ef4444", "高风险"), ("#f59e0b", "中风险"), ("#22c55e", "低风险")])
    return m


def dispatch_map(plans: pd.DataFrame) -> folium.Map:
    m = base_map()
    if plans.empty:
        return m
    for row in plans.itertuples():
        folium.Marker(
            [row.source_lat, row.source_lng],
            tooltip=f"调出 {row.source_grid_id}",
            icon=folium.Icon(color="red", icon="arrow-up"),
        ).add_to(m)
        folium.Marker(
            [row.target_lat, row.target_lng],
            tooltip=f"调入 {row.target_grid_id}",
            icon=folium.Icon(color="green", icon="arrow-down"),
        ).add_to(m)
        folium.PolyLine(
            locations=[[row.source_lat, row.source_lng], [row.target_lat, row.target_lng]],
            color="blue",
            weight=3,
            opacity=0.7,
            popup=f"{row.dispatch_bikes} 辆，{row.distance_km} km，优先级 {row.priority}",
        ).add_to(m)
    add_map_legend(m, "调度路线说明", [("#ef4444", "调出区域"), ("#22c55e", "调入区域"), ("#2563eb", "建议调度路线")])
    return m


def demand_prediction_map(predictions: pd.DataFrame) -> folium.Map:
    """Map shortage-risk grid areas; high demand_gap points are emphasized."""
    m = base_map()
    if predictions.empty:
        return m
    df = predictions.copy()
    if "demand_gap" not in df.columns:
        df["demand_gap"] = pd.to_numeric(df["predicted_start_count"], errors="coerce").fillna(0) - pd.to_numeric(df["predicted_end_count"], errors="coerce").fillna(0)
    for row in df.itertuples():
        area_name, landmark_distance, area_type = nearest_area_name(row.center_lng, row.center_lat)
        risk = getattr(row, "risk_level", "低")
        gap = float(getattr(row, "demand_gap", 0) or 0)
        color = "red" if risk == "高" else ("orange" if risk == "中" else "lightgray")
        radius = 8 if risk == "高" else (6 if risk == "中" else 4)
        fill_opacity = 0.86 if risk == "高" else (0.62 if risk == "中" else 0.28)
        folium.CircleMarker(
            location=[row.center_lat, row.center_lng],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=2 if risk == "高" else 1,
            tooltip=f"{area_name}：{risk}缺车风险，缺口 {gap:.2f}",
            popup=(
                f"网格区域: {area_name}<br>类型: {area_type}<br>距参考地标: {landmark_distance} km<br>"
                f"grid_id: {row.grid_id}<br>预测时间: {row.predict_time}<br>"
                f"预测借车需求: {row.predicted_start_count}<br>预测还车供给: {row.predicted_end_count}<br>"
                f"供需缺口: {gap:.2f}<br>风险等级: {risk}"
            ),
        ).add_to(m)
    add_map_legend(m, "缺车风险等级", [("#ef4444", "高风险网格区域"), ("#f59e0b", "中风险网格区域"), ("#d1d5db", "低风险/平衡区域")])
    return m


def dispatch_route_map(plans: pd.DataFrame) -> folium.Map:
    """Dispatch route map with high-priority tasks highlighted."""
    m = base_map()
    if plans.empty:
        return m
    for row in plans.itertuples():
        high = str(row.priority) == "高"
        line_color = "#dc2626" if high else "#2563eb"
        line_weight = 5 if high else 3
        folium.Marker(
            [row.source_lat, row.source_lng],
            tooltip=f"调出网格 {row.source_grid_id}",
            icon=folium.Icon(color="red", icon="arrow-up"),
        ).add_to(m)
        folium.Marker(
            [row.target_lat, row.target_lng],
            tooltip=f"调入网格 {row.target_grid_id}",
            icon=folium.Icon(color="green", icon="arrow-down"),
        ).add_to(m)
        folium.PolyLine(
            locations=[[row.source_lat, row.source_lng], [row.target_lat, row.target_lng]],
            color=line_color,
            weight=line_weight,
            opacity=0.82,
            popup=f"建议调度 {row.dispatch_bikes} 辆，距离 {row.distance_km} km，优先级 {row.priority}",
        ).add_to(m)
    add_map_legend(m, "调度路线说明", [("#ef4444", "高优先级路线/调出区"), ("#2563eb", "普通路线"), ("#22c55e", "调入区域")])
    return m


def weather_map(weather_df: pd.DataFrame) -> folium.Map:
    m = base_map(zoom_start=11)
    if weather_df.empty:
        return m
    latest = weather_df.sort_values("date").iloc[-1]
    temp = latest.get("temperature_mean", "")
    rain = latest.get("precipitation_sum", "")
    wind = latest.get("wind_speed_max", "")
    holiday = latest.get("holiday_name", "") or ("周末" if int(latest.get("is_weekend", 0) or 0) else "工作日")
    weather_code = int(latest.get("weather_code", 0) or 0)
    icon = "cloud" if weather_code >= 3 else "sun"
    if weather_code >= 51 or float(rain or 0) > 0:
        icon = "tint"
    min_lng, min_lat, max_lng, max_lat = liangshan_bounds()
    temperature = float(temp or 0)
    wind_value = float(wind or 0)
    rain_value = float(rain or 0)
    color = "#2c7bb6"
    if temperature >= 30:
        color = "#d7191c"
    elif temperature >= 24:
        color = "#fdae61"
    elif temperature >= 16:
        color = "#abdda4"
    if rain_value > 0:
        color = "#3288bd"

    rows = 8
    cols = 10
    for i in range(rows):
        for j in range(cols):
            lat1 = min_lat + (max_lat - min_lat) * i / rows
            lat2 = min_lat + (max_lat - min_lat) * (i + 1) / rows
            lng1 = min_lng + (max_lng - min_lng) * j / cols
            lng2 = min_lng + (max_lng - min_lng) * (j + 1) / cols
            opacity = 0.10 + 0.18 * ((i + j) % 4) / 3
            folium.Rectangle(
                bounds=[[lat1, lng1], [lat2, lng2]],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                weight=0,
            ).add_to(m)

    arrow = "➤" if wind_value < 20 else ("➜" if wind_value < 30 else "➠")
    arrow_size = 16 if wind_value < 20 else (21 if wind_value < 30 else 25)
    for i in range(3):
        for j in range(4):
            lat = min_lat + (max_lat - min_lat) * (i + 1) / 4
            lng = min_lng + (max_lng - min_lng) * (j + 1) / 5
            folium.Marker(
                [lat, lng],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="font-size:{arrow_size}px;color:#0f172a;font-weight:700;
                                text-shadow:0 1px 3px rgba(255,255,255,.9);
                                transform: rotate(35deg);">{arrow}</div>
                    <div style="font-size:11px;color:#0f172a;font-weight:700;
                                background:rgba(255,255,255,.75);border-radius:4px;padding:1px 3px;">{wind} km/h</div>
                    """
                ),
                tooltip=f"最大风速 {wind} km/h（风向为示意）",
            ).add_to(m)

    folium.Marker(
        location=[LIANGSHAN_CENTER[1], LIANGSHAN_CENTER[0]],
        icon=folium.Icon(color="blue", icon=icon, prefix="fa"),
        tooltip=f"梁山县天气：{temp}°C，降水 {rain}mm",
        popup=(
            f"日期: {latest.get('date')}<br>"
            f"平均气温: {temp} °C<br>"
            f"降水量: {rain} mm<br>"
            f"最大风速: {wind} km/h<br>"
            f"天气代码: {weather_code}<br>"
            f"节假日: {holiday}"
        ),
    ).add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,.92); border:1px solid #cbd5e1;
                border-radius:8px; padding:10px 12px; font-size:13px;
                box-shadow:0 8px 18px rgba(15,23,42,.16);">
      <b>梁山县气象图层</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#2c7bb6;"></span> 低温/降水
      <span style="display:inline-block;width:12px;height:12px;background:#abdda4;margin-left:8px;"></span> 适宜
      <span style="display:inline-block;width:12px;height:12px;background:#fdae61;margin-left:8px;"></span> 偏热
      <span style="display:inline-block;width:12px;height:12px;background:#d7191c;margin-left:8px;"></span> 高温<br>
      箭头大小表示最大风速强弱；Open-Meteo 当前数据未提供风向，方向仅作示意
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m
