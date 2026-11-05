import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.storyboard_artist import (
    StoryboardArtist,
    VisDescDecompositionResponse,
    system_prompt_template_design_storyboard,
)
from interfaces import CharacterInScene, ShotBriefDescription


class _RecordingDecompositionChain:
    def __init__(self):
        self.input = None

    def __or__(self, _other):
        return self

    async def ainvoke(self, input):
        self.input = input
        return VisDescDecompositionResponse(
            ff_desc="start",
            ff_vis_char_idxs=[0],
            lf_desc="end",
            lf_vis_char_idxs=[0],
            motion_desc="move",
            variation_type="small",
            variation_reason="constraint",
        )


class StoryboardArtistRequirementBindingTests(unittest.IsolatedAsyncioTestCase):
    def test_storyboard_prompt_allows_multiple_scenes_and_requires_explicit_setting(self):
        prompt = system_prompt_template_design_storyboard.casefold()

        self.assertNotIn("only one scene", prompt)
        self.assertIn("one or more scenes", prompt)
        self.assertIn("visual_desc must explicitly state the visible setting", prompt)

    async def test_decomposition_binds_current_shot_index_and_requirement_only_to_chain_input(self):
        chain = _RecordingDecompositionChain()
        prompt_messages = []
        brief = ShotBriefDescription(
            idx=1,
            is_last=False,
            cam_idx=0,
            visual_desc="Mira turns toward the lantern.",
            audio_desc="",
        )
        characters = [CharacterInScene(
            idx=0,
            identifier_in_scene="Mira",
            is_visible=True,
            static_features="blue coat",
            dynamic_features="turns",
        )]
        requirement = "Shot 1 must be medium or large."

        def make_chain(messages):
            prompt_messages.extend(messages)
            return chain

        with patch("agents.storyboard_artist.ChatPromptTemplate.from_messages", side_effect=make_chain):
            shot = await StoryboardArtist(object()).decompose_visual_description(
                brief,
                characters,
                user_requirement=requirement,
            )

        self.assertEqual(chain.input["shot_idx"], 1)
        self.assertEqual(chain.input["user_requirement_str"], requirement)
        self.assertIn("Preserve in both ff_desc and lf_desc", prompt_messages[0][1])
        self.assertNotIn("user_requirement", shot.model_dump())

    async def test_decomposition_receives_audio_verbatim_through_real_prompt_formatting(self):
        audio = '普通话；画外声音说：“等一下，编号是 {A-2}。”随后保持安静。'
        brief = ShotBriefDescription(
            idx=0, is_last=True, cam_idx=0,
            visual_desc="二维动画。女性拿起手机，听见呼喊后停住。",
            audio_desc=audio,
        )
        characters = [CharacterInScene(
            idx=0, identifier_in_scene="Mira", is_visible=True,
            static_features="短发", dynamic_features="米色毛衣",
        )]
        calls = []

        async def respond(prompt):
            calls.append(prompt.to_messages())
            return AIMessage(content=VisDescDecompositionResponse(
                ff_desc="手机在桌面。", ff_vis_char_idxs=[0],
                lf_desc="右手握手机停在胸前。", lf_vis_char_idxs=[0],
                motion_desc="拿起手机，听见画外呼喊后停住，闭口倾听。",
                variation_type="small", variation_reason="拿起手机。",
            ).model_dump_json())

        shot = await StoryboardArtist(RunnableLambda(respond)).decompose_visual_description(
            brief, characters, user_requirement="二维动画，保留安静结尾。",
        )

        self.assertEqual(len(calls), 1)
        self.assertIn(f"<AUDIO_DESC>\n{audio}\n</AUDIO_DESC>", calls[0][1].content)
        self.assertIn(brief.visual_desc, calls[0][1].content)
        self.assertEqual(shot.audio_desc, audio)
        self.assertEqual(shot.motion_desc, "拿起手机，听见画外呼喊后停住，闭口倾听。")


if __name__ == "__main__":
    unittest.main()
