import tempfile
import unittest
from pathlib import Path

from agents.continuity_image_retriever import ImageRetrievalResult, RetrievedImage
from agents.frame_prompt_writer import FinalFramePrompt, FramePromptWriter, _SYSTEM_PROMPT
from continuity.frame_material_context import FrameMaterialContext, history_sample_purpose
from continuity.history_materials import (
    HistoricalSampleCandidate,
    HistoricalSampleGroup,
    StateFieldLineage,
)
from continuity.timeline import ObservedShotSource


class Model:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = []

    def __or__(self, parser):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        return FinalFramePrompt(prompt="Draw Mira at the desk.")


class WriterTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_combines_target_with_known_state(self):
        prompt = " ".join(_SYSTEM_PROMPT.split())
        self.assertIn("target description and known field state jointly define", prompt)
        self.assertIn("explicit target instructions take priority", prompt)
        self.assertNotIn("target alone determines", prompt)

    async def test_history_sample_is_described_as_actual_image_not_field_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.png"
            path.write_bytes(b"history")
            source = ObservedShotSource(shot_idx=1)
            lineage = [StateFieldLineage(
                entity_id="character-0",
                field="position",
                value="desk",
                source=source,
            )]
            sample = HistoricalSampleCandidate(
                image_id="history",
                asset_path=str(path),
                time_seconds=2.0,
            )
            group = HistoricalSampleGroup(
                source=source,
                lineage=lineage,
                samples=[sample],
            )
            context = FrameMaterialContext(
                shot_idx=2,
                frame_type="first",
                fixed_materials=[],
                field_lineage=lineage,
                history_sample_groups=[group],
                portrait_candidates=[],
                required_slots=0,
                top_k=1,
            )
            result = ImageRetrievalResult(images=[RetrievedImage(
                candidate_id="history",
                purpose=history_sample_purpose(group, sample),
            )])
            model = Model()

            await FramePromptWriter(model).write("Mira waits at the desk.", context, result)

            system = model.messages[0].content
            self.assertIn("not proof of every field", system)
            self.assertNotIn("prove only their stated fields", system)

    async def test_history_sample_rejects_a_purpose_that_binds_it_to_state_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.png"
            path.write_bytes(b"history")
            source = ObservedShotSource(shot_idx=1)
            lineage = [StateFieldLineage(
                entity_id="character-0",
                field="position",
                value="desk",
                source=source,
            )]
            sample = HistoricalSampleCandidate(
                image_id="history",
                asset_path=str(path),
                time_seconds=2.0,
            )
            group = HistoricalSampleGroup(
                source=source,
                lineage=lineage,
                samples=[sample],
            )
            context = FrameMaterialContext(
                shot_idx=2,
                frame_type="first",
                fixed_materials=[],
                field_lineage=lineage,
                history_sample_groups=[group],
                portrait_candidates=[],
                required_slots=0,
                top_k=1,
            )
            result = ImageRetrievalResult(images=[RetrievedImage(
                candidate_id="history",
                purpose="source shot 1 proves character-0.position=desk",
            )])

            with self.assertRaisesRegex(ValueError, "invalid purpose"):
                await FramePromptWriter(Model()).write("Mira waits at the desk.", context, result)


if __name__ == "__main__":
    unittest.main()
