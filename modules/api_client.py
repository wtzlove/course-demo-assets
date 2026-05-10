import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests
from requests import Response
from requests.exceptions import ReadTimeout, RequestException

from config.settings import RAW_DIR, SD_API_URL, SD_CLIENT_ID, SD_CLIENT_SECRET


DEFAULT_TIMEOUT = (10, 60)
DEFAULT_SLEEP_SECONDS = 0.8
DEFAULT_RETRIES = 2


def _hash_user(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sanitize_order(order: dict[str, Any]) -> dict[str, Any]:
    safe = dict(order)
    safe.pop("phone", None)
    user_id = safe.pop("userNewId", None)
    if user_id not in (None, ""):
        safe["user_hash"] = _hash_user(user_id)
    return safe


def _signature(client_id: str, client_secret: str, timestamp: str, nonce: str) -> str:
    sign_text = client_id + timestamp + nonce
    digest = hmac.new(
        client_secret.encode("utf-8"),
        sign_text.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def build_headers() -> dict[str, str]:
    if not SD_CLIENT_ID or not SD_CLIENT_SECRET:
        raise RuntimeError("未配置 SD_CLIENT_ID 或 SD_CLIENT_SECRET，请在 .env 中填写。")
    timestamp = str(int(time.time() * 1000))
    nonce = uuid.uuid4().hex
    return {
        "X-Client-Id": SD_CLIENT_ID,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": _signature(SD_CLIENT_ID, SD_CLIENT_SECRET, timestamp, nonce),
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0",
    }


def build_request_url(page_no: int, start_date: str, end_date: str) -> str:
    """Match the standalone PyCharm crawler: params are appended to URL, not sent as data."""
    params = {
        "pageNo": str(page_no),
        "startDate": start_date,
        "endDate": end_date,
    }
    query_string = urlencode(params, quote_via=quote)
    return f"{SD_API_URL}?{query_string}"


def decode_response(response_text: str) -> dict[str, Any]:
    outer = json.loads(response_text)
    inner_raw = outer.get("data") if isinstance(outer, dict) else None
    if isinstance(inner_raw, str):
        try:
            outer["inner_data"] = json.loads(inner_raw)
        except json.JSONDecodeError:
            outer["inner_data"] = {}
    elif isinstance(inner_raw, dict):
        outer["inner_data"] = inner_raw
    else:
        outer["inner_data"] = {}
    return outer


def extract_order_list(decoded_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively collect every orderList field to tolerate small response shape changes."""
    orders: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            order_list = obj.get("orderList")
            if isinstance(order_list, list):
                orders.extend([item for item in order_list if isinstance(item, dict)])
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(decoded_json.get("inner_data") or decoded_json)
    return orders


def _post_with_retry(url: str, timeout: tuple[int, int] = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES) -> Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=build_headers(), timeout=timeout)
            response.raise_for_status()
            return response
        except ReadTimeout as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except RequestException:
            raise
    raise RuntimeError(f"接口请求失败：{last_error}")


def fetch_page(
    page_no: int,
    start_date: str,
    end_date: str,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    url = build_request_url(page_no, start_date, end_date)
    response = _post_with_retry(url, timeout=timeout, retries=retries)
    decoded = decode_response(response.text)
    return {
        "page_no": page_no,
        "decoded": decoded,
        "orders": extract_order_list(decoded),
        "text": response.text,
        "request_url": response.url,
        "status_code": response.status_code,
    }


def fetch_range(
    start_date: str,
    end_date: str,
    max_pages: int = 5,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> dict[str, Any]:
    all_orders: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    for page_no in range(1, max_pages + 1):
        page = fetch_page(page_no, start_date, end_date)
        pages.append(page)
        all_orders.extend(page["orders"])
        if not page["orders"]:
            break
        if sleep_seconds > 0 and page_no < max_pages:
            time.sleep(sleep_seconds)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(RAW_DIR) / f"api_orders_{stamp}.json"
    backup_path.write_text(
        json.dumps(
            {
                "start_date": start_date,
                "end_date": end_date,
                "pages": len(pages),
                "orders": [_sanitize_order(order) for order in all_orders],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"orders": all_orders, "pages": len(pages), "backup_path": str(backup_path)}
