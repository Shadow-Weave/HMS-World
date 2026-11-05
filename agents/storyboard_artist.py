from typing import List, Optional, Literal
import asyncio
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from interfaces import CharacterInScene, ShotDescription, ShotBriefDescription

from utils.retry import after_func



system_prompt_template_design_storyboard = \
"""
[Task]
Turn <SCRIPT>, which may contain one or more scenes, into separately generated shots that edit together. <CHARACTERS> defines the cast; <USER_REQUIREMENT> defines global and shot-specific constraints. Preserve the source plot and scene order.

[Sequence]
- Give each shot one narrative purpose within a cause-and-effect progression. A reveal must be an observable event.
- Allocate time and shots by dramatic weight: let discovery or hesitation breathe, compress routine passages, and split overloaded exchanges at meaningful responses.
- Motivate each cut through an action, gaze, reaction, reveal, or scripted time/place change. Carry relevant prop ownership, action progress, and emotional residue across continuous shots. Ellipsis and montage may bridge unshown changes.
- Reuse a camera position when it serves the next beat; change it to reveal necessary information. Maintain the action axis where continuity depends on it, distinguishing physical geography from screen position after a reverse angle.
- Honor requested shot counts and runtime. Use relative pacing unless timing is supplied for an individual shot; a whole-film duration is not a per-shot duration.

[visual_desc]
Every visual_desc must explicitly state the visible setting or location, including its distinguishing environment and light. Then describe:
1. Camera and treatment: framing, angle, movement, and concrete line/shadow, material, or camera-handling properties appropriate to the requested style. Keep this treatment coherent across related shots.
2. Entry state: visible participants, placement, facing, identifying features, and relevant object positions. Restate necessary context rather than using "same as before".
3. Action: one goal with the steps that accomplish it. For interactions, specify contact, grip, support, transfer, and release where needed; keep object counts coherent.
4. Performance: perception leads to processing and a behavioral response. Show gaze, posture, or gesture rather than listing emotions or directing every facial muscle.
5. Exit state: the position, held objects, or unresolved action needed for the next cut. Continuing movement is valid; do not force a freeze or replay the completed action in the following shot.
Describe only visible content; off-screen events belong in audio_desc.

[audio_desc]
- This field alone contains verbatim dialogue: preserve words and language, label the speaker and delivery, distinguish visible speech, off-screen speech, and narration. In visual_desc, describe when the person speaks or reacts without repeating the words.
- Tie sound effects to their triggers and maintain the ambience of a continuous location.
- Preserve intended pauses, silence, music, and voice-over; do not fill gaps with invented speech or simultaneous talking.

[Output]
Use the script's language for values and the following schema:
{format_instructions}
"""


human_prompt_template_design_storyboard = \
"""
<SCRIPT>
{script_str}
</SCRIPT>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""



system_prompt_template_decompose_visual_description = \
"""
[Task and inputs]
Decompose <VISUAL_DESC> into static boundary frames and motion for one continuous take. <AUDIO_DESC> supplies sound/dialogue; <CHARACTERS> supplies identity. Apply global <USER_REQUIREMENT> constraints and only those specific to <SHOT_INDEX>.

[Shared constraints]
- Preserve the planned narrative purpose, staging, and camera assignment; do not add events, people, props, cuts, or reference inputs.
- Preserve in both ff_desc and lf_desc the requested visual medium and style properties, and carry them into motion_desc. Use compatible line/shadow or material/light vocabulary rather than mixing 2D, 3D, and live-action defaults.
- Use relative pacing unless timing is explicitly supplied for this shot. A whole-film runtime is not its duration.

[Boundary frames]
- ff_desc and lf_desc describe visible snapshots: framing, layout, light, connected body pose, facing/gaze, and prop support. Express actual posture rather than an intention such as "about to stand"; keep anatomical and crop requirements compatible.
- Capture the action's entry and exit phases, including movement already in progress. A boundary does not require a restart or freeze.

[motion_desc]
The video model receives this field and audio_desc with frame images, without the storyboard or user requirements. Write the visual instruction in this order:
1. Camera behavior: fixed or moving, distinct from subject movement.
2. Starting state: identify participants by stable visible features rather than bare names or internal tags. Declare identity, costume, and object-count constraints once, allowing scripted changes.
3. Action beats: one main goal per beat, connected causally. Include the necessary physical contact and transfer of support between reaching, gripping, lifting, placing, or passing an object.
4. Performance: link a visual or audio stimulus to attention, processing, and a motivated response. Use speech cues to align the performance; exact spoken words remain solely in audio_desc. Request lip-sync for visible speakers while listeners do not mouth off-screen voices or narration. Retain intentional silence.
5. Ending state: specify position, orientation, prop ownership, and performance matching lf_desc, even when only a first-frame image will be supplied. Add only take-specific exclusions, not a generic negative list.

[Variation]
Classify change within this take, subject to explicit shot-specific requirements:
- small: existing participants change expression, pose, or position; moderate pan, tilt, or tracking.
- medium: a new participant enters or someone turns from back view to front view.
- large: substantial camera travel or a major shift in composition/focus, such as wide to close-up.

[Output]
Use the script's language for values and the following schema:
{format_instructions}
"""


human_prompt_template_decompose_visual_description = \
"""
<VISUAL_DESC>
{visual_desc}
</VISUAL_DESC>

<AUDIO_DESC>
{audio_desc}
</AUDIO_DESC>

<SHOT_INDEX>
{shot_idx}
</SHOT_INDEX>

<CHARACTERS>
{characters_str}
</CHARACTERS>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""


class VisDescDecompositionResponse(BaseModel):
    ff_desc: str = Field(
        description='Opening frame.',
    )
    ff_vis_char_idxs: List[int] = Field(
        description='Indices of characters visible in the opening frame, using the supplied character-list order.',
    )
    lf_desc: str = Field(
        description='Closing frame.',
    )
    lf_vis_char_idxs: List[int] = Field(
        description='Indices of characters visible in the closing frame, using the supplied character-list order.',
    )
    motion_desc: str = Field(
        description='Visual movement instruction for the video generator.',
    )
    variation_type: Literal['large', 'medium', 'small'] = Field(
        description='Degree of visual change within the take.',
    )
    variation_reason: str = Field(
        description='Reason for the selected variation type.',
    )



class StoryboardArtist:
    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = chat_model


    @retry(stop=stop_after_attempt(3), after=after_func)
    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: Optional[str] = None,
        retry_timeout: int = 150,
    ) -> List[ShotBriefDescription]:

        class StoryboardResponse(BaseModel):
            storyboard: List[ShotBriefDescription] = Field(
                description="A complete storyboard of the scene, including the visual and audio description of each shot.",
            )

        script_str = script.strip()
        characters_str = "\n".join([f"Character {index}: {char}" for index, char in enumerate(characters)])
        user_requirement_str = user_requirement.strip() if user_requirement else ""

        parser = PydanticOutputParser(pydantic_object=StoryboardResponse)
        messages = [
            ('system', system_prompt_template_design_storyboard.format(format_instructions=parser.get_format_instructions())),
            ('human', human_prompt_template_design_storyboard.format(script_str=script_str, characters_str=characters_str, user_requirement_str=user_requirement_str)),
        ]
        chain = self.chat_model | parser
        response: StoryboardResponse = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        storyboard = response.storyboard

        return storyboard




    @retry(stop=stop_after_attempt(3), after=after_func)
    async def decompose_visual_description(
        self,
        shot_brief_desc: ShotBriefDescription,
        characters: List[CharacterInScene],
        user_requirement: Optional[str] = None,
        retry_timeout: int = 150,
    ) -> ShotDescription:
        parser = PydanticOutputParser(pydantic_object=VisDescDecompositionResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt_template_decompose_visual_description),
                ('human', human_prompt_template_decompose_visual_description),
            ]
        )
        chain = prompt_template | self.chat_model | parser

        visual_desc = shot_brief_desc.visual_desc.strip()

        characters_str = "\n".join([f"{char.identifier_in_scene}: (static) {char.static_features}; (dynamic) {char.dynamic_features}" for char in characters])
        user_requirement_str = user_requirement.strip() if user_requirement else ""

        decomposition: VisDescDecompositionResponse = await asyncio.wait_for(
            chain.ainvoke(
                input={
                    "format_instructions": parser.get_format_instructions(),
                    "visual_desc": visual_desc,
                    "audio_desc": shot_brief_desc.audio_desc,
                    "shot_idx": shot_brief_desc.idx,
                    "characters_str": characters_str,
                    "user_requirement_str": user_requirement_str,
                },
            ),
            timeout=retry_timeout,
        )

        validate_char_idxs(decomposition.ff_vis_char_idxs, len(characters), "ff_vis_char_idxs")
        validate_char_idxs(decomposition.lf_vis_char_idxs, len(characters), "lf_vis_char_idxs")

        return ShotDescription(
            idx=shot_brief_desc.idx,
            is_last=shot_brief_desc.is_last,
            cam_idx=shot_brief_desc.cam_idx,
            visual_desc=shot_brief_desc.visual_desc,
            variation_type=decomposition.variation_type,
            variation_reason=decomposition.variation_reason,
            ff_desc=decomposition.ff_desc,
            ff_vis_char_idxs=decomposition.ff_vis_char_idxs,
            lf_desc=decomposition.lf_desc,
            lf_vis_char_idxs=decomposition.lf_vis_char_idxs,
            motion_desc=decomposition.motion_desc,
            audio_desc=shot_brief_desc.audio_desc,
        )


def validate_char_idxs(idxs, num_characters, field_name):
    """Reject LLM-emitted character indices outside [0, num_characters).

    Negative values would silently select the wrong character via Python
    indexing; out-of-range values would crash deep inside the render gather.
    Raising here lets the @retry on decompose_visual_description re-ask.
    """
    invalid = [idx for idx in idxs if idx < 0 or idx >= num_characters]
    if invalid:
        raise ValueError(
            f"{field_name} contains invalid character indices {invalid}; "
            f"valid range is 0..{num_characters - 1}"
        )
