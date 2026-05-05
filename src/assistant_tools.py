"""Assistant tool functions registered with LiteRT-LM."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


WEATHER_CODE_TEXT = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
}


def _fetch_json(url: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_current_time(timezone_name: str = "local") -> str:
    """Get the current date and time.

    Args:
        timezone_name: IANA timezone name such as Asia/Shanghai or America/Los_Angeles. Use local for the server timezone.
    """
    if not timezone_name or timezone_name == "local":
        now = datetime.now().astimezone()
    else:
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError:
            return f"无法识别时区：{timezone_name}。请使用 IANA 时区名称，例如 Asia/Shanghai。"

    tz = now.tzname() or timezone_name
    return now.strftime(f"%Y-%m-%d %H:%M:%S {tz}")


def get_weather(location: str | None = None) -> str:
    """Get current weather for a city or place.

    Args:
        location: City or place name, for example 上海, 北京, Tokyo, or San Francisco.
    """
    location = (location or os.environ.get("WEATHER_DEFAULT_LOCATION") or "").strip()
    if not location:
        return "请先告诉我城市或地点，例如：上海天气怎么样？"

    try:
        place = _geocode_location(location)
        if place is None:
            return f"没有找到地点：{location}。请换一个更明确的城市名。"

        weather = _fetch_current_weather(place["latitude"], place["longitude"])
    except Exception as exc:
        return f"天气查询失败：{exc}"

    current = weather.get("current", {})
    units = weather.get("current_units", {})
    city = place.get("name", location)
    country = place.get("country", "")
    admin = place.get("admin1", "")
    place_text = "，".join(part for part in [country, admin, city] if part)
    weather_code = current.get("weather_code")
    weather_text = WEATHER_CODE_TEXT.get(weather_code, f"天气代码 {weather_code}")

    temp_unit = units.get("temperature_2m", "C")
    wind_unit = units.get("wind_speed_10m", "km/h")
    precipitation_unit = units.get("precipitation", "mm")
    humidity_unit = units.get("relative_humidity_2m", "%")

    return (
        f"{place_text} 当前天气：{weather_text}，"
        f"气温 {current.get('temperature_2m')}{temp_unit}，"
        f"体感 {current.get('apparent_temperature')}{temp_unit}，"
        f"湿度 {current.get('relative_humidity_2m')}{humidity_unit}，"
        f"降水 {current.get('precipitation')}{precipitation_unit}，"
        f"风速 {current.get('wind_speed_10m')}{wind_unit}。"
    )


def _geocode_location(location: str) -> dict | None:
    params = urllib.parse.urlencode({
        "name": location,
        "count": 1,
        "language": "zh",
        "format": "json",
    })
    data = _fetch_json(f"https://geocoding-api.open-meteo.com/v1/search?{params}")
    results = data.get("results") or []
    return results[0] if results else None


def _fetch_current_weather(latitude: float, longitude: float) -> dict:
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ]),
        "timezone": "auto",
        "forecast_days": 1,
    })
    return _fetch_json(f"https://api.open-meteo.com/v1/forecast?{params}")
