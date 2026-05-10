from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler


_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
    return _scheduler


def start_scheduler(job_func, interval_minutes: int = 30) -> str:
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
    if scheduler.get_job("bike_crawl_job"):
        scheduler.remove_job("bike_crawl_job")
    scheduler.add_job(job_func, "interval", minutes=interval_minutes, id="bike_crawl_job", next_run_time=datetime.now())
    return f"定时采集已启动，每 {interval_minutes} 分钟执行一次。"


def stop_scheduler() -> str:
    scheduler = get_scheduler()
    if scheduler.get_job("bike_crawl_job"):
        scheduler.remove_job("bike_crawl_job")
    return "定时采集已停止。"


def recent_days_range(days: int = 3) -> tuple[str, str]:
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")
