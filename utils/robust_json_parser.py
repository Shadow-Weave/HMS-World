# Final structured answers must contain complete JSON. Preserve the existing
# repair for models that emit trailing commas, without accepting partial drafts.
import json
import re
from typing import List

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


class TrailingCommaTolerantPydanticOutputParser(PydanticOutputParser):
    """Require complete JSON for final answers, retaining trailing-comma repair.

    LangChain's default partial-JSON recovery can silently close an unfinished
    draft before a later complete answer. Final answers must be valid as a
    whole, or contain exactly one complete JSON code block.
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False):
        if partial or not result:
            return super().parse_result(result, partial=partial)
        text = result[0].text
        candidates = [text]
        fences = _JSON_FENCE_RE.findall(text)
        if len(fences) == 1:
            candidates.append(fences[0])
        for candidate in candidates:
            for value in (candidate, strip_trailing_commas(candidate)):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                return self._parse_obj(parsed)
        raise OutputParserException(
            "Expected a complete JSON answer or one complete JSON code block; "
            "unfinished drafts are not accepted.",
            llm_output=text,
        )
