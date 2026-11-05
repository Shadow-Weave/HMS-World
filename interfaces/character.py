from pydantic import BaseModel, Field
from typing import Optional


class CharacterInScene(BaseModel):
    idx: int = Field(
        description="The index of the character in the scene, starting from 0",
    )
    identifier_in_scene: str = Field(
        description="The identifier for the character in this specific scene, which may differ from the base identifier",
        examples=["Alice", "Bob the Builder"],
    )
    is_visible: bool = Field(
        description="Indicates whether the character is visible in this scene",
        examples=[True, False],
    )
    static_features: str = Field(
        description="The static features of the character in this specific scene, such as facial features and body shape that remain constant or are rarely changed. If the character is not visible, this field can be left empty.",
        examples=[
            "Alice has long blonde hair and blue eyes, and is of slender build.",
            "Bob the Builder is a middle-aged man with a sturdy build.",
        ]
    )
    dynamic_features: Optional[str] = Field(
        default=None,
        description="The dynamic features of the character in this specific scene, such as clothing and accessories that may change from scene to scene. If not mentioned, this field can be left empty. If the character is not visible, this field should be None.",
        examples=[
            "Wearing a red scarf and a black leather jacket",
        ]
    )

    def __str__(self):
        # Alice[visible]
        # static features: Alice has long blonde hair and blue eyes, and is of slender build.
        # dynamic features: Wearing a red scarf and a black leather jacket

        s = f"{self.identifier_in_scene}"
        s += "[visible]" if self.is_visible else "[not visible]"
        s += "\n"
        s += f"static features: {self.static_features}\n"
        s += f"dynamic features: {self.dynamic_features}\n"

        return s
