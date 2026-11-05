import copy
import hashlib
import unittest
from unittest.mock import patch

from agents.continuity_image_retriever import (
    ContinuityImageRetriever,
    FrameSelections,
    PortraitSelection,
    _SYSTEM_PROMPT,
    _sample_history_context,
    _selection_message,
    _validate_frame_selections,
)
from continuity.frame_material_context import (
    FixedFrameMaterial,
    FrameMaterialContext,
    IdentityPortraitCandidate,
    NativeFrameCandidate,
    VisibleCharacterPortraitCandidates,
)
from continuity.history_materials import (
    HistoricalSampleCandidate,
    HistoricalSampleGroup,
    StateFieldLineage,
)
from continuity.timeline import ObservedShotSource


class Model:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def __or__(self, parser):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.response


def portrait_candidates(character_id: str, suffix: str = "") -> VisibleCharacterPortraitCandidates:
    return VisibleCharacterPortraitCandidates(
        character_id=character_id,
        candidates=[
            IdentityPortraitCandidate(
                image_id=f"front{suffix}", asset_path="/front", view="front"
            ),
            IdentityPortraitCandidate(
                image_id=f"side{suffix}", asset_path="/side", view="side"
            ),
            IdentityPortraitCandidate(
                image_id=f"back{suffix}", asset_path="/back", view="back"
            ),
        ],
    )


def history_group(shot_idx: int, *image_ids: str) -> HistoricalSampleGroup:
    source = ObservedShotSource(shot_idx=shot_idx)
    return HistoricalSampleGroup(
        source=source,
        lineage=[StateFieldLineage(
            entity_id="character-0",
            field="position",
            value="desk",
            source=source,
        )],
        samples=[
            HistoricalSampleCandidate(
                image_id=image_id,
                asset_path=f"/{image_id}",
                time_seconds=float(index),
            )
            for index, image_id in enumerate(image_ids)
        ],
    )


def context(
    *,
    portraits: list[VisibleCharacterPortraitCandidates] | None = None,
    groups: list[HistoricalSampleGroup] | None = None,
    native: list[NativeFrameCandidate] | None = None,
    top_k: int = 4,
) -> FrameMaterialContext:
    portraits = portraits or []
    return FrameMaterialContext(
        shot_idx=2,
        frame_type="first",
        fixed_materials=[],
        field_lineage=[item for group in groups or [] for item in group.lineage],
        history_sample_groups=groups or [],
        portrait_candidates=portraits,
        native_candidates=native or [],
        required_slots=len(portraits),
        top_k=top_k,
    )


class RetrieverTests(unittest.IsolatedAsyncioTestCase):
    @patch("agents.continuity_image_retriever.image_path_to_b64", return_value="image")
    async def test_opt_in_correction_receives_error_and_preserves_candidates(self, _image):
        invalid = FrameSelections(selections=[], history_sample_ids=["a", "b"])
        valid = FrameSelections(selections=[], history_sample_ids=["b"])

        class SequenceModel(Model):
            def __init__(self):
                super().__init__(None)
                self.messages = []

            async def ainvoke(self, messages):
                self.messages.append(messages)
                self.calls += 1
                return invalid if self.calls == 1 else valid

        model = SequenceModel()
        retriever = ContinuityImageRetriever(model)
        retriever.correct_invalid_selection_once = True
        material_context = context(groups=[history_group(1, "a", "b")])
        before = material_context.model_dump()
        result = await retriever.retrieve(material_context, "Target")
        self.assertEqual([item.candidate_id for item in result.images], ["b"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.messages[0], model.messages[1][:2])
        self.assertEqual(model.messages[1][2].content, invalid.model_dump_json())
        self.assertIn("at most one history sample", model.messages[1][3].content)
        self.assertEqual(material_context.model_dump(), before)

    @patch("agents.continuity_image_retriever.image_path_to_b64", return_value="image")
    async def test_correction_keeps_strict_validation_and_stops_after_one_retry(self, _image):
        model = Model(FrameSelections(selections=[], history_sample_ids=["a", "b"]))
        retriever = ContinuityImageRetriever(model)
        retriever.correct_invalid_selection_once = True
        with self.assertRaisesRegex(ValueError, "at most one history sample"):
            await retriever.retrieve(context(groups=[history_group(1, "a", "b")]), "Target")
        self.assertEqual(model.calls, 2)

    @patch("agents.continuity_image_retriever.image_path_to_b64", return_value="image")
    async def test_user_history_exception_matches_exact_target_and_selection(self, _image):
        material_context = context(groups=[history_group(1, "a", "b")])
        selections = FrameSelections(selections=[], history_sample_ids=["a", "b"])
        retriever = ContinuityImageRetriever(Model(selections))
        retriever.same_source_history_approvals = [{
            "shot_idx": 2, "frame_type": "first",
            "target_sha256": hashlib.sha256(b"Target").hexdigest(),
            "selections": selections.model_dump(), "reason": "Explicit user exception",
        }]
        result = await retriever.retrieve(material_context, "Target")
        self.assertEqual([item.candidate_id for item in result.images], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "at most one history"):
            await retriever.retrieve(material_context, "Different target")
        with self.assertRaisesRegex(ValueError, "remaining capacity"):
            await retriever.retrieve(material_context.model_copy(update={"top_k": 1}), "Target")

    def test_history_offers_tail_and_one_stable_random_frame_per_source(self):
        group = history_group(1, *(f"sample-{index}" for index in range(10)))
        material_context = context(groups=[group])

        first = _sample_history_context(material_context)
        second = _sample_history_context(material_context)
        offered = [sample.image_id for sample in first.history_sample_groups[0].samples]

        self.assertEqual(first, second)
        self.assertEqual(len(offered), 2)
        self.assertEqual(offered[-1], "sample-9")
        self.assertIn(offered[0], {f"sample-{index}" for index in range(9)})
        self.assertEqual(len(material_context.history_sample_groups[0].samples), 10)

    def test_candidate_labels_use_only_real_ids(self):
        material_context = context(
            groups=[history_group(1, "shot-1-sample-frame-120")],
            native=[NativeFrameCandidate(
                candidate_id="native-2-first-0", asset_path="/native", description="draft"
            )],
        )
        with patch("agents.continuity_image_retriever.image_path_to_b64", return_value="image"):
            content = _selection_message("Mira waits.", material_context)

        self.assertEqual(content[1]["text"], "History sample id=shot-1-sample-frame-120; source shot 1; time=0s")
        self.assertEqual(content[3]["text"], "Native candidate id=native-2-first-0; description=draft")
        self.assertEqual([item["type"] for item in content[1:]], ["text", "image_url", "text", "image_url"])

    def test_prompt_states_lineage_boundary_and_reserved_fixed_materials(self):
        prompt = " ".join(_SYSTEM_PROMPT.split())
        self.assertIn("target description and state lineage jointly define", prompt)
        self.assertIn("locates a source shot, not a particular sample", prompt)
        self.assertIn("does not prove that every sample shows the lineage value", prompt)
        self.assertIn("Fixed materials are already reserved", prompt)

        material_context = context().model_copy(update={
            "fixed_materials": [FixedFrameMaterial(
                image_id="first-frame-2",
                asset_path="/first-frame-2",
                role="current_first_frame_starting_state",
            )],
            "required_slots": 1,
        })
        text = _selection_message("Mira waits.", material_context)[0]["text"]
        self.assertIn("<RESERVED_FIXED_MATERIAL_ROLES>", text)
        self.assertIn("current shot first-frame starting state", text)

    async def test_zero_capacity_without_portraits_returns_fixed_without_model_call(self):
        material_context = context(groups=[history_group(1, "history-a")], top_k=0)
        model = Model(None)

        result = await ContinuityImageRetriever(model).retrieve(material_context, "Empty room.")

        self.assertEqual(model.calls, 0)
        self.assertEqual(result.images, [])

    async def test_no_selectable_candidates_without_portraits_skips_model(self):
        material_context = context(top_k=3)
        model = Model(None)

        result = await ContinuityImageRetriever(model).retrieve(material_context, "Empty room.")

        self.assertEqual(model.calls, 0)
        self.assertEqual(result.images, [])

    async def test_required_material_capacity_is_checked_before_any_model_call(self):
        material_context = context(portraits=[portrait_candidates("character-0")], top_k=0)
        model = Model(None)

        with self.assertRaisesRegex(ValueError, "required slots exceed"):
            await ContinuityImageRetriever(model).retrieve(material_context, "Mira waits.")

        self.assertEqual(model.calls, 0)

    def test_portrait_selection_rejects_unknown_cross_character_duplicate_and_out_of_order(self):
        material_context = context(
            portraits=[
                portrait_candidates("character-0", "-zero"),
                portrait_candidates("character-1", "-one"),
            ],
        )
        invalid_selections = (
            (
                "unknown",
                [
                    PortraitSelection(character_id="character-0", image_id="missing"),
                    PortraitSelection(character_id="character-1", image_id="side-one"),
                ],
                "not a candidate",
            ),
            (
                "cross-character",
                [
                    PortraitSelection(character_id="character-0", image_id="side-one"),
                    PortraitSelection(character_id="character-1", image_id="side-one"),
                ],
                "not a candidate",
            ),
            (
                "duplicate",
                [
                    PortraitSelection(character_id="character-0", image_id="front-zero"),
                    PortraitSelection(character_id="character-1", image_id="front-zero"),
                ],
                "must not reuse",
            ),
            (
                "out-of-order",
                [
                    PortraitSelection(character_id="character-1", image_id="side-one"),
                    PortraitSelection(character_id="character-0", image_id="side-zero"),
                ],
                "preserve visible-character order",
            ),
        )

        duplicate_context = context(
            portraits=[
                portrait_candidates("character-0", "-zero"),
                VisibleCharacterPortraitCandidates(
                    character_id="character-1",
                    candidates=[
                        IdentityPortraitCandidate(
                            image_id="front-zero", asset_path="/shared", view="front"
                        ),
                        IdentityPortraitCandidate(
                            image_id="side-one", asset_path="/side", view="side"
                        ),
                        IdentityPortraitCandidate(
                            image_id="back-one", asset_path="/back", view="back"
                        ),
                    ],
                ),
            ],
        )
        for label, selections, message in invalid_selections:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_frame_selections(
                        duplicate_context if label == "duplicate" else material_context,
                        FrameSelections(selections=selections),
                    )

    def test_history_selection_rejects_unknown_duplicate_and_two_samples_from_one_shot(self):
        material_context = context(groups=[history_group(1, "history-a", "history-b")])
        invalid_ids = (
            (["missing"], "Unknown history"),
            (["history-a", "history-a"], "not be selected more than once"),
            (["history-a", "history-b"], "at most one history sample"),
        )

        for sample_ids, message in invalid_ids:
            with self.subTest(sample_ids=sample_ids):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_frame_selections(
                        material_context,
                        FrameSelections(selections=[], history_sample_ids=sample_ids),
                    )

    def test_native_selection_rejects_unknown_duplicate_and_preserves_input_order(self):
        native = [
            NativeFrameCandidate(candidate_id="native-a", asset_path="/a", description="a"),
            NativeFrameCandidate(candidate_id="native-b", asset_path="/b", description="b"),
        ]
        material_context = context(native=native)
        for ids, message in (
            (["missing"], "Unknown native"),
            (["native-a", "native-a"], "not be selected more than once"),
        ):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_frame_selections(
                        material_context,
                        FrameSelections(selections=[], native_candidate_ids=ids),
                    )

        _, _, selected = _validate_frame_selections(
            material_context,
            FrameSelections(selections=[], native_candidate_ids=["native-b", "native-a"]),
        )
        self.assertEqual([item.candidate_id for item in selected], ["native-a", "native-b"])

    def test_history_and_native_share_one_optional_capacity_and_do_not_mutate_context(self):
        material_context = context(
            groups=[history_group(1, "history-a")],
            native=[NativeFrameCandidate(candidate_id="native-a", asset_path="/a", description="a")],
            top_k=1,
        )
        original = copy.deepcopy(material_context)

        with self.assertRaisesRegex(ValueError, "exceeds remaining capacity"):
            _validate_frame_selections(
                material_context,
                FrameSelections(
                    selections=[],
                    history_sample_ids=["history-a"],
                    native_candidate_ids=["native-a"],
                ),
            )

        self.assertEqual(material_context, original)


if __name__ == "__main__":
    unittest.main()
