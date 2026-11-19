"""Offline locale-resource contracts; no application, camera or C++ build needed."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "RayTrophiStudio/assets/i18n"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate translation key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def placeholders(value):
    printf = re.findall(r"%[-+ #0-9.*]*(hh|ll|[hlLjzt])?([diuoxXfFeEgGaAcspn])",
                        value.replace("%%", ""))
    named = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", value))
    return printf, named


class CatalogContracts(unittest.TestCase):
    def test_registry_and_complete_turkish_coverage(self):
        registry = read(ASSETS / "locales.json")
        codes = [entry["code"] for entry in registry["locales"]]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertIn("en", codes)
        self.assertIn("tr", codes)
        english = read(ASSETS / "en.json")
        self.assertGreater(len(english), 150)
        for entry in registry["locales"]:
            self.assertRegex(entry["code"], r"^[a-z]+(?:-[a-z]+)*$")
            self.assertTrue(entry["name"])
            translated = read(ASSETS / (entry["code"] + ".json"))
            self.assertEqual(set(english), set(translated), entry["code"])
            for key, value in translated.items():
                self.assertIsInstance(value, str)
                self.assertTrue(value.strip(), key)
                self.assertNotIn("###", value, key)
                self.assertEqual(placeholders(english[key]), placeholders(value), key)

    def test_source_keys_and_fallbacks_match_english_catalog(self):
        english = read(ASSETS / "en.json")
        pattern = re.compile(r'\btr(?:Id)?\(\s*"([^"\n]+)"\s*,\s*("(?:\\.|[^"\\])*")\s*\)')
        sources = ROOT / "RayTrophiStudio/source/src/UI"
        count = 0
        for source in sources.glob("PlayerOperator*UI.cpp"):
            for match in pattern.finditer(source.read_text(encoding="utf-8-sig")):
                key, fallback = match.group(1), json.loads(match.group(2))
                self.assertIn(key, english, str(source))
                self.assertEqual(english[key], fallback, key)
                count += 1
        self.assertGreater(count, 150)

    def test_turkish_text_and_named_placeholders_are_preserved(self):
        turkish = read(ASSETS / "tr.json")
        self.assertEqual(turkish["player.game_library"], "Oyun kütüphanesi")
        self.assertEqual(turkish["player.save"], "Kaydet")
        self.assertEqual(turkish["player.tracked_count"], "{count} oyuncu takip ediliyor")
        self.assertTrue(set("çğıİöşü") <= set("".join(turkish.values())))


if __name__ == "__main__":
    unittest.main()
