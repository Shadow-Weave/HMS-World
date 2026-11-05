from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Literal, Tuple


class ShotBriefDescription(BaseModel):
    idx: int = Field(
        description='Zero-based shot index.',
    )
    is_last: bool = Field(
        description='Whether this shot ends the script.',
    )

    # visual
    cam_idx: int = Field(
        description='Camera index in the scene.',
    )
    visual_desc: str = Field(
        description='Visual progression. Use angle-bracketed character-list identifiers, e.g. <Alice>.',
    )


    # audio
    audio_desc: str = Field(
        description='Soundscape and spoken lines.',
    )

    def __str__(self):
        s = f"Shot {self.idx}:\n"
        s += f"Camera Index: {self.cam_idx}\n"
        s += f"Visual: {self.visual_desc}\n"
        if self.audio_desc:
            s += f"Audio: {self.audio_desc}"
        return s


class ShotDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idx: int = Field(
        description='Zero-based shot index.',
    )
    is_last: bool = Field(
        description='Whether this shot ends the script.',
    )

    # visual
    cam_idx: int = Field(
        description='Camera index in the scene.',
    )
    visual_desc: str = Field(
        description='Visual progression. Use angle-bracketed character-list identifiers, e.g. <Alice>.',
    )
    variation_type: Literal['large', 'medium', 'small'] = Field(
        description='Degree of visual change within the take.',
    )
    variation_reason: str = Field(
        description='Reason for the selected variation type.',
    )

    ff_desc: str = Field(
        description='Opening frame.',
    )
    ff_vis_char_idxs: List[int] = Field(
        default=[],
        description='Indices of characters visible in the opening frame.',
    )
    lf_desc: str = Field(
        description='Closing frame.',
    )
    lf_vis_char_idxs: List[int] = Field(
        default=[],
        description='Indices of characters visible in the closing frame.',
    )
    motion_desc: str = Field(
        description='Visual movement instruction for the video generator.',
    )

    # audio
    audio_desc: str = Field(
        description='Soundscape and spoken lines.',
    )
