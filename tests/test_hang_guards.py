"""Bounded download retries and timeout behavior."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from utils.image import download_image
from utils.video import download_video


def _no_sleep(fn):
    retrying = getattr(fn, "retry", None)
    if retrying is not None:
        retrying.sleep = lambda seconds: None


class TestDownloadRetries(unittest.TestCase):
    def setUp(self):
        _no_sleep(download_image)
        _no_sleep(download_video)

    def test_download_image_gives_up_on_persistent_network_error(self):
        calls = {"n": 0}

        def flaky_get(url, **kwargs):
            calls["n"] += 1
            if calls["n"] < 10:
                raise requests.ConnectionError("connection refused")
            return MagicMock()

        with patch("utils.image.requests.get", side_effect=flaky_get):
            with self.assertRaises(requests.ConnectionError):
                download_image("http://example.com/a.png", "/tmp/a.png")
        self.assertLessEqual(calls["n"], 5, "retry must be bounded, not retry-until-success")

    def test_download_image_does_not_retry_client_errors(self):
        calls = {"n": 0}
        gone = MagicMock()
        gone.raise_for_status.side_effect = requests.HTTPError(
            "404", response=MagicMock(status_code=404)
        )

        def expired_then_ok(url, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                return gone
            return MagicMock()

        with patch("utils.image.requests.get", side_effect=expired_then_ok):
            with self.assertRaises(requests.HTTPError):
                download_image("http://example.com/expired.png", "/tmp/a.png")
        self.assertEqual(calls["n"], 1, "4xx responses must fail fast, not be retried")

    def test_download_image_sets_a_timeout(self):
        captured = {}

        def record_get(url, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("utils.image.requests.get", side_effect=record_get):
            download_image("http://example.com/a.png", "/tmp/a.png")
        self.assertIsNotNone(captured.get("timeout"), "requests.get must not wait forever")

    def test_download_video_gives_up_on_persistent_network_error(self):
        calls = {"n": 0}

        def flaky_get(url, **kwargs):
            calls["n"] += 1
            if calls["n"] < 10:
                raise requests.ConnectionError("connection refused")
            return MagicMock()

        with patch("utils.video.requests.get", side_effect=flaky_get):
            with self.assertRaises(requests.ConnectionError):
                download_video("http://example.com/a.mp4", "/tmp/a.mp4")
        self.assertLessEqual(calls["n"], 5, "retry must be bounded, not retry-until-success")


if __name__ == "__main__":
    unittest.main()
