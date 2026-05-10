import requests
import tomllib

from config.settings import BASE_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from modules.database import query_df, system_summary
from modules.geo_context import nearest_area_name


def get_llm_config() -> dict:
    config = {"api_key": LLM_API_KEY, "base_url": LLM_BASE_URL, "model": LLM_MODEL}
    config_path = BASE_DIR / "llm_config.toml"
    if config_path.exists():
        try:
            file_config = tomllib.loads(config_path.read_text(encoding="utf-8")).get("llm", {})
            config.update({k: v for k, v in file_config.items() if v})
        except Exception:
            pass
    return config


def build_context(engine=None) -> dict:
    s = system_summary(engine)
    latest_model = query_df("SELECT * FROM model_logs ORDER BY id DESC LIMIT 1", engine=engine)
    dispatch = query_df("SELECT COUNT(*) AS tasks, COALESCE(SUM(dispatch_bikes), 0) AS bikes FROM dispatch_plans", engine=engine)
    s["model_metrics"] = latest_model.iloc[0].to_dict() if not latest_model.empty else {}
    s["dispatch_tasks"] = int(dispatch["tasks"].iloc[0]) if not dispatch.empty else 0
    s["top_hotspots"] = _hotspot_context(engine)
    s["prediction_risks"] = _prediction_context(engine)
    s["dispatch_plans"] = _dispatch_context(engine)
    s["weather_summary"] = _weather_context(engine)
    return s


def _area(lng, lat) -> str:
    try:
        name, distance, area_type = nearest_area_name(float(lng), float(lat))
        return f"{name}（{area_type}，距参考地标{distance}km）"
    except Exception:
        return "未知区域"


def _hotspot_context(engine=None) -> list[dict]:
    df = query_df(
        """
        SELECT h.grid_id, h.stat_time, h.end_count, h.hotspot_level, h.hotspot_score,
               h.center_lng, h.center_lat, COALESCE(g.start_count, 0) AS start_count
        FROM hotspot_results h
        LEFT JOIN grid_hour_stats g ON h.grid_id=g.grid_id AND h.stat_time=g.stat_time
        ORDER BY h.hotspot_score DESC
        LIMIT 10
        """,
        engine=engine,
    )
    if df.empty:
        return []
    df["area"] = df.apply(lambda row: _area(row["center_lng"], row["center_lat"]), axis=1)
    return df[["area", "grid_id", "stat_time", "start_count", "end_count", "hotspot_level", "hotspot_score"]].to_dict("records")


def _prediction_context(engine=None) -> list[dict]:
    df = query_df(
        """
        SELECT grid_id, predict_time, predicted_start_count, predicted_end_count,
               risk_level, center_lng, center_lat
        FROM prediction_results
        ORDER BY predicted_end_count DESC
        LIMIT 10
        """,
        engine=engine,
    )
    if df.empty:
        return []
    df["area"] = df.apply(lambda row: _area(row["center_lng"], row["center_lat"]), axis=1)
    df["gap_start_minus_end"] = (df["predicted_start_count"] - df["predicted_end_count"]).round(2)
    return df[["area", "grid_id", "predict_time", "predicted_start_count", "predicted_end_count", "gap_start_minus_end", "risk_level"]].to_dict("records")


def _dispatch_context(engine=None) -> list[dict]:
    df = query_df(
        """
        SELECT source_grid_id, target_grid_id, source_lng, source_lat, target_lng, target_lat,
               dispatch_bikes, distance_km, priority, reason
        FROM dispatch_plans
        ORDER BY dispatch_bikes DESC, distance_km ASC
        LIMIT 10
        """,
        engine=engine,
    )
    if df.empty:
        return []
    df["source_area"] = df.apply(lambda row: _area(row["source_lng"], row["source_lat"]), axis=1)
    df["target_area"] = df.apply(lambda row: _area(row["target_lng"], row["target_lat"]), axis=1)
    return df[["source_area", "target_area", "dispatch_bikes", "distance_km", "priority", "reason"]].to_dict("records")


def _weather_context(engine=None) -> list[dict]:
    df = query_df(
        """
        SELECT w.date, w.temperature_mean, w.precipitation_sum, w.wind_speed_max, w.weather_code,
               COALESCE(h.is_weekend, 0) AS is_weekend,
               COALESCE(h.is_holiday, 0) AS is_holiday,
               COALESCE(h.holiday_name, '') AS holiday_name
        FROM weather_daily w
        LEFT JOIN holiday_calendar h ON w.date=h.date
        ORDER BY w.date DESC
        LIMIT 7
        """,
        engine=engine,
    )
    if df.empty:
        return []
    df["date_type"] = df.apply(lambda row: row["holiday_name"] if row["is_holiday"] else ("周末" if row["is_weekend"] else "工作日"), axis=1)
    return df[["date", "temperature_mean", "precipitation_sum", "wind_speed_max", "weather_code", "date_type"]].to_dict("records")


def local_rule_answer(question: str, context: dict) -> str:
    q = question.strip()
    if any(k in q for k in ["热点", "为什么"]):
        hotspots = context.get("top_hotspots", [])
        detail = ""
        if hotspots:
            top = hotspots[0]
            detail = (
                f"当前排名最高的是 {top['area']}，统计时间 {top['stat_time']}，"
                f"开始订单数量 {top['start_count']}，结束订单数量 {top['end_count']}，热点得分 {top['hotspot_score']}。"
            )
        return (
            f"当前系统识别到高等级热点 {context.get('hotspot_high', 0)} 个。{detail}"
            "热点主要依据网格小时终点订单量判定，"
            "终点订单越集中，说明该区域停放压力越明显。若该区域周边存在商圈、学校、医院或交通节点，"
            "则更容易在短时间内形成共享单车集中停放。"
        )
    if any(k in q for k in ["预测", "模型", "准确"]):
        metrics = context.get("model_metrics", {})
        return (
            f"当前预测高风险区域 {context.get('prediction_high', 0)} 个。系统优先使用随机森林模型，数据量不足时使用历史均值模型兜底。"
            f"最近模型样本数为 {metrics.get('sample_count', 0) or 0}，MAE={metrics.get('mae', '暂无')}，"
            f"RMSE={metrics.get('rmse', '暂无')}，R2={metrics.get('r2', '暂无')}。提高准确率的关键是持续积累更多不同日期和小时的订单样本。"
        )
    if any(k in q for k in ["调度", "方案", "车辆"]):
        plans = context.get("dispatch_plans", [])
        detail = ""
        if plans:
            top = plans[0]
            detail = f"当前优先方案是从 {top['source_area']} 调往 {top['target_area']}，建议调度 {top['dispatch_bikes']} 辆，距离 {top['distance_km']}km。"
        return (
            f"当前调度任务数为 {context.get('dispatch_tasks', 0)}，建议调动车辆 {context.get('dispatch_bikes', 0)} 辆。"
            f"{detail}"
            "source_grid_id 表示车辆过剩区域，target_grid_id 表示车辆短缺区域，dispatch_bikes 表示建议调度数量，"
            "distance_km 表示调度距离。优先级越高，越适合运维人员优先处理。"
        )
    if any(k in q for k in ["运营", "建议"]):
        return (
            "根据当前预测结果，建议运维人员优先关注终点订单集中且风险等级较高的网格，对车辆堆积明显区域进行清理或外调。"
            "同时，对预测借车需求较高但车辆供给不足的区域，可在高峰前提前补车，减少用户找车成本。"
        )
    if any(k in q for k in ["论文", "段落", "分析"]):
        return (
            "从系统运行结果来看，共享单车订单在空间上呈现明显集聚特征，终点热点主要集中在县城核心区域及公共服务设施周边。"
            "通过网格化统计与短时预测模型，系统能够识别下一时段可能出现停放压力的区域，并结合供需差异生成调度建议，"
            "为共享单车精细化管理提供辅助决策依据。"
        )
    if any(k in q for k in ["数据量", "足够"]):
        return (
            f"当前清洗后有效订单 {context.get('clean_orders', 0)} 条。若仅用于系统流程验证，少量数据即可完成演示；"
            "若用于较稳定的短时预测，建议持续采集多个日期、覆盖早晚高峰和周末样本，使每个热点网格具备更多小时级历史记录。"
        )
    return (
        "我可以解释热点、预测结果、调度方案，也可以生成运营建议和论文分析段落。"
        f"当前系统累计有效订单 {context.get('clean_orders', 0)} 条，高等级热点 {context.get('hotspot_high', 0)} 个。"
    )


def external_llm_answer(question: str, context: dict) -> str | None:
    config = get_llm_config()
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    model = config.get("model", "")
    if not (api_key and base_url and model):
        return None
    prompt = (
        "你是共享单车停放热点预测系统的中文分析助手。只能使用聚合统计信息回答，不能要求或输出手机号、用户ID、原始订单明细。\n"
        "请优先引用以下真实系统结果：top_hotspots、prediction_risks、dispatch_plans、weather_summary、model_metrics。"
        "回答要说明依据，不要只重复模型名称。\n"
        f"系统上下文：{context}\n用户问题：{question}"
    )
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "用简洁中文回答，面向本科毕设答辩演示。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def answer_question(question: str, engine=None) -> str:
    context = build_context(engine)
    llm_answer = external_llm_answer(question, context)
    return llm_answer or local_rule_answer(question, context)
