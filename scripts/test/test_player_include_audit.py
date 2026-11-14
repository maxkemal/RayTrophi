"""Regression checks for first-line SDK includes in the Player audit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "audit_player_includes.py"
SPEC = importlib.util.spec_from_file_location("player_include_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PlayerIncludeAuditTests(unittest.TestCase):
    def includes(self, text, encoding):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.cpp"
            path.write_text(text, encoding=encoding)
            return AUDIT.parse_includes(str(path))

    def test_bom_first_line_keeps_sdk_dependency(self):
        text = '#include <embree4/rtcore.h>\n#include "Shared.h"\n'
        expected = [("embree4/rtcore.h", True), ("Shared.h", False)]
        for encoding in ("utf-8", "utf-8-sig"):
            with self.subTest(encoding=encoding):
                found = self.includes(text, encoding)
                self.assertEqual(found, expected)
                self.assertEqual(AUDIT.classify_external(found[0][0]), "embree")

    def test_bom_first_line_keeps_player_condition(self):
        text = (
            "#ifdef RT_NO_CUDA\n"
            '#include "Host.h"\n'
            "#else\n"
            "#include <cuda_runtime.h>\n"
            "#endif\n"
        )
        self.assertEqual(self.includes(text, "utf-8-sig"), [("Host.h", False)])


if __name__ == "__main__":
    unittest.main()
