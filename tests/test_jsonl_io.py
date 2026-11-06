"""Regression checks for single-writer generation-file appends."""

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.io import append_jsonl, read_jsonl


class AppendJsonlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "generations.jsonl"

    def test_new_file_creates_parent_and_round_trips(self):
        append_jsonl(self.path, {"id": 0})
        self.assertEqual(read_jsonl(self.path), [{"id": 0}])
        self.assertEqual(self.path.read_bytes(), b'{"id": 0}\n')

    def test_empty_existing_file_has_no_leading_blank_line(self):
        self.path.parent.mkdir()
        self.path.touch()
        append_jsonl(self.path, {"id": 0})
        self.assertEqual(self.path.read_bytes(), b'{"id": 0}\n')

    def test_resume_without_final_newline_preserves_records(self):
        self.path.parent.mkdir()
        original = b'{"id": 0}\n{"id": 1}'
        self.path.write_bytes(original)
        self.assertEqual(read_jsonl(self.path), [{"id": 0}, {"id": 1}])
        append_jsonl(self.path, {"id": 2})
        append_jsonl(self.path, {"id": 3})
        self.assertEqual(read_jsonl(self.path), [{"id": i} for i in range(4)])
        self.assertEqual(self.path.read_bytes(), original + b'\n{"id": 2}\n{"id": 3}\n')

    def test_existing_line_endings_are_preserved(self):
        self.path.parent.mkdir()
        for ending in (b"\n", b"\r\n"):
            with self.subTest(ending=ending):
                original = b'{"id": 0}' + ending
                self.path.write_bytes(original)
                append_jsonl(self.path, {"id": 1})
                self.assertEqual(self.path.read_bytes(), original + b'{"id": 1}\n')
                self.assertEqual(read_jsonl(self.path), [{"id": 0}, {"id": 1}])

    def test_utf8_tail_is_not_decoded_as_a_partial_character(self):
        self.path.parent.mkdir()
        # Trailing Unicode whitespace is accepted by read_jsonl's strip().
        original = '{"id": 0, "text": "推理"}\u3000'.encode("utf-8")
        self.path.write_bytes(original)
        append_jsonl(self.path, {"id": 1, "text": "答案"})
        self.assertEqual(self.path.read_bytes(), original + '\n{"id": 1, "text": "答案"}\n'.encode("utf-8"))
        self.assertEqual(read_jsonl(self.path), [{"id": 0, "text": "推理"}, {"id": 1, "text": "答案"}])

    def test_serialization_failure_does_not_modify_existing_file(self):
        self.path.parent.mkdir()
        original = b'{"id": 0}'
        self.path.write_bytes(original)
        with self.assertRaises(TypeError):
            append_jsonl(self.path, {"invalid": object()})
        self.assertEqual(self.path.read_bytes(), original)

    def test_truncated_json_is_not_silently_repaired(self):
        self.path.parent.mkdir()
        original = b'{"id":'
        self.path.write_bytes(original)
        append_jsonl(self.path, {"id": 1})
        self.assertTrue(self.path.read_bytes().startswith(original))
        with self.assertRaises(json.JSONDecodeError):
            read_jsonl(self.path)


if __name__ == "__main__":
    unittest.main()
