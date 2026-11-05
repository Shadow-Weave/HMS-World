"""Agent LLM error handling and session persistence regression tests."""

import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock


from agent_runtime.llm import OpenAICompatibleLLM
from agent_runtime.loop import AgentLoop
from agent_runtime.prompts import PromptBuilder
from agent_runtime.session_index import SessionIndex
from agent_runtime.tool_executor import ToolExecutor
from agent_runtime.tools import ToolRegistry


class FakeStatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"http status {status_code}")


def _fake_completion(text="ok"):
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    message.model_dump.return_value = {}
    return MagicMock(choices=[MagicMock(message=message)])


class TestLLMClient(unittest.IsolatedAsyncioTestCase):
    def _llm(self, create):
        llm = OpenAICompatibleLLM(model="m", base_url="http://localhost:1", api_key="k")
        llm.client = MagicMock(chat=MagicMock(completions=MagicMock(create=create)))
        return llm

    async def test_retries_rate_limit_then_succeeds(self):
        create = AsyncMock(side_effect=[FakeStatusError(429), _fake_completion("recovered")])
        llm = self._llm(create)
        result = await llm.complete([{"role": "user", "content": "x"}], tools=[])
        self.assertEqual(result.text, "recovered")
        self.assertEqual(create.await_count, 2)

    async def test_does_not_retry_auth_errors(self):
        create = AsyncMock(side_effect=FakeStatusError(401))
        llm = self._llm(create)
        with self.assertRaises(FakeStatusError):
            await llm.complete([{"role": "user", "content": "x"}], tools=[])
        self.assertEqual(create.await_count, 1)

    async def test_gives_up_after_bounded_attempts(self):
        create = AsyncMock(side_effect=FakeStatusError(500))
        llm = self._llm(create)
        with self.assertRaises(FakeStatusError):
            await llm.complete([{"role": "user", "content": "x"}], tools=[])
        self.assertLessEqual(create.await_count, 4)
        self.assertGreater(create.await_count, 1)

    async def test_empty_choices_raises_clear_error(self):
        create = AsyncMock(return_value=MagicMock(choices=[]))
        llm = self._llm(create)
        with self.assertRaisesRegex(RuntimeError, "choice"):
            await llm.complete([{"role": "user", "content": "x"}], tools=[])


class BoomLLM:
    async def complete(self, messages, tools):
        raise RuntimeError("boom-llm")


class TestLoopErrorBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_llm_failure_emits_error_and_persists_failed_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), BoomLLM())
            events = [event async for event in loop.stream_events("hi")]
            kinds = [event["type"] for event in events]
            self.assertIn("error", kinds)
            error_event = next(event for event in events if event["type"] == "error")
            self.assertIn("boom-llm", error_event["message"])
            self.assertEqual(events[-2]["type"], "done")
            self.assertEqual(events[-1]["type"], "session")
            active = index.active()
            records = index.get(active["session_id"])["recent_turn_records"]
            self.assertEqual(records[-1]["status"], "failed")


class TestSessionIndexDurability(unittest.TestCase):
    def test_corrupt_sessions_file_is_backed_up_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            index.create(idea="precious work", session_id="keep-me")
            index.sessions_path.write_text("{ definitely not json", encoding="utf-8")
            data = index.load()
            self.assertEqual(data["sessions"], {})
            backups = list(index.state_dir.glob("sessions.json.corrupt-*"))
            self.assertEqual(len(backups), 1, "corrupt state must be preserved for recovery, not discarded")
            self.assertIn("definitely not json", backups[0].read_text(encoding="utf-8"))

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            index.create(session_id="roundtrip")
            self.assertEqual(list(index.state_dir.glob("*.tmp")), [])
            self.assertIn("roundtrip", index.load()["sessions"])

    def test_concurrent_creates_do_not_lose_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_a = SessionIndex(tmp)
            index_b = SessionIndex(tmp)

            def worker(index, tag):
                for i in range(40):
                    index.create(session_id=f"s-{tag}-{i}")

            threads = [
                threading.Thread(target=worker, args=(index_a, "a")),
                threading.Thread(target=worker, args=(index_b, "b")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            sessions = index_a.load()["sessions"]
            self.assertEqual(len(sessions), 80, "concurrent read-modify-write must not lose sessions")


if __name__ == "__main__":
    unittest.main()
