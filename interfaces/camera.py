from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Tuple



class Camera(BaseModel):
    idx: int = Field(
        description="The index of the camera in the scene, starting from 0.",
    )

    active_shot_idxs: List[int] = Field(
        description="The indices of the shots that the camera can film.",
    )

    parent_cam_idx: Optional[int] = Field(
        default=None,
        description="The index of the parent camera. If the camera has no parent, set this to None.",
    )

    parent_shot_idx: Optional[int] = Field(
        default=None,
        description="The index of the dependent shot. If the camera has no parent, set this to None.",
    )

    reason: Optional[str] = Field(
        default=None,
        description="The reason for the selection of the parent camera. If the camera has no parent, set this to None.",
    )

    is_parent_fully_covers_child: Optional[bool] = Field(
        default=None,
        description="Whether the parent camera fully covers the child camera's content. If the camera has no parent, set this to None.",
    )

    missing_info: Optional[str] = Field(
        default=None,
        description="The missing information in the child shot that is not covered by the parent shot. If the parent shot fully covers the child shot, set this to None.",
    )


def validate_serial_camera_parents(cameras: List[Camera]) -> None:
    """A serial reference must already exist when its child first appears."""
    by_idx = {camera.idx: camera for camera in cameras}
    for camera in cameras:
        if not camera.active_shot_idxs:
            raise ValueError(f"Serial camera {camera.idx} has no active shots")
        if (camera.parent_cam_idx is None) != (camera.parent_shot_idx is None):
            raise ValueError(f"Serial camera {camera.idx} has an incomplete parent binding")
        if camera.parent_cam_idx is None:
            continue
        parent = by_idx.get(camera.parent_cam_idx)
        if parent is None or camera.parent_shot_idx not in parent.active_shot_idxs:
            raise ValueError(f"Serial camera {camera.idx} parent shot is not on its parent camera")
        if camera.parent_shot_idx >= min(camera.active_shot_idxs):
            raise ValueError(f"Serial camera {camera.idx} requires a strictly earlier parent shot")
