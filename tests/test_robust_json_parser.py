import unittest

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser


class Items(BaseModel):
    items: list[int]


class CompleteJSONParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = TrailingCommaTolerantPydanticOutputParser(pydantic_object=Items)

    def test_parses_complete_plain_and_fenced_answers(self):
        for text in ['{"items":[1,2]}', '```json\n{"items":[1,2]}\n```']:
            with self.subTest(text=text):
                self.assertEqual(self.parser.parse(text).items, [1, 2])

    def test_retains_trailing_comma_support(self):
        self.assertEqual(self.parser.parse('{"items":[1,2,],}').items, [1, 2])

    def test_rejects_unfinished_answer_instead_of_silently_closing_it(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse('{"items":[1,2')

    def test_uses_complete_fenced_answer_instead_of_recovering_partial_draft(self):
        text = '{"items":[1,2\nI will start again.\n```json\n{"items":[1,2,3,4,5,6,7,8]}\n```'
        self.assertEqual(self.parser.parse(text).items, list(range(1, 9)))

    def test_does_not_choose_between_multiple_json_answers(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse('```json\n{"items":[1]}\n```\n```json\n{"items":[2]}\n```')
