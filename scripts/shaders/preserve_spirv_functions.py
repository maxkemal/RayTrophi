"""Preserve non-entry SPIR-V function boundaries before exhaustive optimization.

Changes only OpFunction's Function Control word: clear Inline, set DontInline.
Does not compile, optimize, renumber IDs, change instructions or deploy shaders.
The result must be checked with spirv-val before and after optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def preserve_functions(data):
    if len(data) < 20 or len(data) % 4:
        raise ValueError('Invalid SPIR-V length')
    words = list(struct.unpack(f'<{len(data) // 4}I', data))
    if words[0] != 0x07230203:
        raise ValueError('Invalid SPIR-V magic')
    entries = set()
    functions = {}
    names = {}
    current = None
    cursor = 5
    while cursor < len(words):
        count, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        if count == 0 or cursor + count > len(words):
            raise ValueError(f'Invalid instruction at word {cursor}')
        if opcode == 15:  # OpEntryPoint: execution model, entry id, name...
            if count < 4:
                raise ValueError('Truncated OpEntryPoint')
            entries.add(words[cursor + 2])
        elif opcode == 5:  # OpName
            if count < 3:
                raise ValueError('Truncated OpName')
            raw = struct.pack(f'<{count - 2}I', *words[cursor + 2:cursor + count])
            names[words[cursor + 1]] = raw.split(b'\0', 1)[0].decode('utf-8', errors='replace')
        elif opcode == 54:  # OpFunction: result type, result id, control, type
            if count != 5 or current is not None:
                raise ValueError('Invalid or nested OpFunction')
            current = words[cursor + 2]
            if current in functions:
                raise ValueError('Duplicate function id')
            functions[current] = cursor
        elif opcode == 56:  # OpFunctionEnd
            if count != 1 or current is None:
                raise ValueError('Unmatched OpFunctionEnd')
            current = None
        cursor += count
    if current is not None or not functions or not entries or not entries <= functions.keys():
        raise ValueError('Missing function/end/entry definition')
    changed = []
    protected = []
    for identifier, offset in functions.items():
        if identifier in entries:
            continue
        control_offset = offset + 3
        old = words[control_offset]
        # SPIR-V Function Control Inline=0x1, DontInline=0x2 are mutually exclusive.
        new = (old & ~1) | 2
        protected.append(identifier)
        if new != old:
            words[control_offset] = new
            changed.append(dict(id=identifier, word_offset=control_offset,
                                name=names.get(identifier, ''), before=old, after=new))
    output = struct.pack(f'<{len(words)}I', *words)
    return output, dict(
        input_sha256=hashlib.sha256(data).hexdigest(),
        output_sha256=hashlib.sha256(output).hexdigest(),
        bytes=len(data), id_bound=words[3], entry_ids=sorted(entries),
        function_count=len(functions), protected_function_count=len(protected),
        changed_function_count=len(changed), changes=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary', action='store_true', help='Omit per-function change details')
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error('--output is required unless --dry-run is used')
    if args.output is not None and args.input.resolve() == args.output.resolve():
        parser.error('Input and output must differ; in-place mutation is forbidden')
    output, report = preserve_functions(args.input.read_bytes())
    if not args.dry_run:
        # A fresh scratch output is required; never overwrite an installed module.
        with args.output.open('xb') as target:
            target.write(output)
    if args.summary:
        report.pop('changes')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
