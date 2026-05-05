import json
import unittest
from unittest.mock import patch

import assistant_tools


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class AssistantToolsTests(unittest.TestCase):
    def test_current_time_rejects_unknown_timezone(self):
        result = assistant_tools.get_current_time("Not/AZone")

        self.assertIn("无法识别时区", result)

    def test_current_time_uses_requested_timezone(self):
        result = assistant_tools.get_current_time("Asia/Shanghai")

        self.assertIn("CST", result)

    def test_weather_asks_for_location_when_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            result = assistant_tools.get_weather("")

        self.assertIn("请先告诉我城市或地点", result)

    def test_weather_returns_current_conditions(self):
        responses = [
            {
                "results": [{
                    "name": "上海",
                    "country": "中国",
                    "admin1": "上海市",
                    "latitude": 31.23,
                    "longitude": 121.47,
                }]
            },
            {
                "current": {
                    "temperature_2m": 18.5,
                    "apparent_temperature": 18.0,
                    "relative_humidity_2m": 60,
                    "precipitation": 0,
                    "weather_code": 2,
                    "wind_speed_10m": 12.3,
                },
                "current_units": {
                    "temperature_2m": "°C",
                    "apparent_temperature": "°C",
                    "relative_humidity_2m": "%",
                    "precipitation": "mm",
                    "wind_speed_10m": "km/h",
                },
            },
        ]

        def fake_urlopen(_url, timeout):
            self.assertEqual(timeout, 8.0)
            return FakeResponse(responses.pop(0))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = assistant_tools.get_weather("上海")

        self.assertIn("中国，上海市，上海 当前天气：局部多云", result)
        self.assertIn("气温 18.5°C", result)
        self.assertIn("湿度 60%", result)

    def test_weather_reports_unknown_location(self):
        with patch("urllib.request.urlopen", lambda _url, timeout: FakeResponse({})):
            result = assistant_tools.get_weather("不存在的地方")

        self.assertIn("没有找到地点", result)


if __name__ == "__main__":
    unittest.main()
