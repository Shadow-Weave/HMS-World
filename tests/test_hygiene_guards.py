"""Rate-limiter locking and package metadata contracts."""

import asyncio
import tomllib
import unittest
from contextlib import suppress
from pathlib import Path

from utils.rate_limiter import RateLimiter

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRateLimiterLocking(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_acquirer_does_not_hold_the_lock(self):
        limiter = RateLimiter(max_requests_per_minute=1)
        await limiter.acquire()  # consume the only slot in the window
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.05)  # waiter is now waiting for the window to free
        try:
            try:
                await asyncio.wait_for(limiter.lock.acquire(), timeout=0.25)
                limiter.lock.release()
            except asyncio.TimeoutError:
                self.fail("rate limiter sleeps while holding its lock, blocking every other caller")
        finally:
            waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter

    async def test_min_delay_smoothing_still_applies(self):
        limiter = RateLimiter(max_requests_per_minute=600)  # min delay 0.1s
        loop = asyncio.get_running_loop()
        start = loop.time()
        await limiter.acquire()
        await limiter.acquire()
        self.assertGreaterEqual(loop.time() - start, 0.08)


class TestPackagingMetadata(unittest.TestCase):
    def test_pyproject_is_consistent(self):
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        readme = data["project"]["readme"]
        self.assertTrue((REPO_ROOT / readme).exists(), f"readme points at missing file: {readme}")
        self.assertNotIn("index", data, "top-level [[index]] is not valid pyproject TOML and is silently ignored")
        dev_group = data.get("dependency-groups", {}).get("dev", [])
        self.assertTrue(any("pytest" in str(item) for item in dev_group), "pytest must be a declared dev dependency so the suite runs from the venv")


if __name__ == "__main__":
    unittest.main()
