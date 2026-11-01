"""No-compiler tests for Function Control-only SPIR-V boundary preservation."""
import importlib.util
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'preserve_spirv_functions', ROOT/'scripts/shaders/preserve_spirv_functions.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def instruction(opcode, *operands):
    return [((len(operands) + 1) << 16) | opcode, *operands]


def module(controls=(0, 1, 4, 8)):
    # Structural fixture, not claimed to be a validator-complete shader.
    words = [0x07230203, 0x10400, 0, 100, 0]
    words += instruction(15, 5316, 10, 0x6e69616d, 0)  # entry main
    offsets = []
    for index, control in enumerate(controls):
        offsets.append(len(words) + 3)
        words += instruction(54, 1, 10 + index, control, 2)
        words += instruction(248, 30 + index)
        words += instruction(253)
        words += instruction(56)
    return struct.pack(f'<{len(words)}I', *words), offsets


class FunctionBoundaries(unittest.TestCase):
    def test_only_nonentry_function_control_words_change(self):
        data, offsets = module()
        output, report = TOOL.preserve_functions(data)
        before = struct.unpack(f'<{len(data)//4}I', data)
        after = struct.unpack(f'<{len(output)//4}I', output)
        self.assertEqual(len(output), len(data))
        self.assertEqual(report['protected_function_count'], 3)
        self.assertEqual(report['entry_ids'], [10])
        self.assertEqual([after[n] for n in offsets], [0, 2, 6, 10])
        differences = {i for i, (a, b) in enumerate(zip(before, after)) if a != b}
        self.assertEqual(differences, set(offsets[1:]))
        self.assertEqual(after[:5], before[:5])

    def test_idempotent_and_names_not_required(self):
        data, _ = module((0, 2, 6, 10))
        output, report = TOOL.preserve_functions(data)
        self.assertEqual(data, output)
        self.assertEqual(report['changed_function_count'], 0)
        self.assertEqual(report['function_count'], 4)
        self.assertEqual(report['input_sha256'], report['output_sha256'])

    def test_invalid_input_is_rejected_before_writing(self):
        data, _ = module()
        fixtures = [b'', data[:-1], b'bad!' + data[4:], data[:-4]]
        words = list(struct.unpack(f'<{len(data)//4}I', data))
        words[5] = 0  # zero word count
        fixtures.append(struct.pack(f'<{len(words)}I', *words))
        for malformed in fixtures:
            with self.subTest(length=len(malformed)):
                with self.assertRaises(ValueError):
                    TOOL.preserve_functions(malformed)

    def test_nonfunction_inline_controls_are_untouched(self):
        data, _ = module()
        words = list(struct.unpack(f'<{len(data)//4}I', data))
        # Function-end is followed by a well-formed non-function instruction.
        words += instruction(0)
        data = struct.pack(f'<{len(words)}I', *words)
        output, _ = TOOL.preserve_functions(data)
        self.assertEqual(output[-4:], data[-4:])


if __name__ == '__main__':
    unittest.main()
