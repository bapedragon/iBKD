from __future__ import annotations

import hashlib
import json
import unittest

from methods.table1_cub200.teacher_contract import (
    TABLE1_TEACHER_BUILD,
    TABLE1_TEACHER_ROOT,
    TABLE1_TEACHER_SHA256,
    validate_table1_teacher_spec,
)


class Table1CUB200TeacherContractTests(unittest.TestCase):
    def test_build543_teacher_is_present_and_hash_locked(self) -> None:
        self.assertEqual(TABLE1_TEACHER_BUILD, 543)
        manifest = json.loads(
            (TABLE1_TEACHER_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        spec = manifest["teachers"]["cub200"]
        validate_table1_teacher_spec(spec)
        checkpoint = TABLE1_TEACHER_ROOT / spec["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.assertEqual(digest, TABLE1_TEACHER_SHA256)

    def test_different_teacher_is_rejected(self) -> None:
        manifest = json.loads(
            (TABLE1_TEACHER_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        spec = dict(manifest["teachers"]["cub200"])
        spec["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "build-543"):
            validate_table1_teacher_spec(spec)


if __name__ == "__main__":
    unittest.main()
