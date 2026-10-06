"""Domain panel field inventory: no editable field is lost in a panel reorganisation.

Every value widget of the simulation domain panel and its helpers is listed with
its label, the expression it writes and the tab it is drawn in. The baseline is
written once, BEFORE a reorganisation:

    python scripts/test/check_domain_panel_fields.py --write-baseline

After it, the default run fails when a baseline widget disappeared without an
entry in CHANGES (moved widgets are fine: the tab is reported, not gated):

    python scripts/test/check_domain_panel_fields.py

A field may be renamed (old label -> new label) or removed on purpose
("removed: <reason>"); both must be written below, so a loss is a decision
that shows up in review, never an accident of moving 4800 lines around.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / 'RayTrophiStudio/source/src/UI'
BASELINE = ROOT / 'docs/dev/domain_panel_fields_baseline.json'

# Files that draw the domain panel. Glob patterns are allowed for helpers added later.
FILES = ['scene_ui_simulation_domains.cpp', 'Matter*Controls.cpp', 'scene_ui_fluid_thermal.cpp',
         'DomainPanel*.cpp']

WIDGET = re.compile(
    r'(?:ImGui|UIWidgets|DomainUi)::'
    r'(Checkbox|Combo|SliderFloat\d?|SliderInt\d?|SliderAngle|DragFloat\d?|DragInt\d?|'
    r'DragFloatRange2|InputFloat\d?|InputInt\d?|InputText|ColorEdit\d|RadioButton|'
    r'Float|Int|Bool|Choice|Color|Slider)'
    r'\s*\(\s*("(?:[^"\\]|\\.)*")\s*,\s*([^,)]*)', re.S)
TAB = re.compile(r'BeginTabItem\s*\(\s*"([^"#]*)')

# Intentional changes: baseline key -> new key, or "removed: reason".
# Key format: "<label without ##id>".
CHANGES = {
    # 2026-10-06 grain reorganisation: material -> Matter tab, solver -> Solvers.
    'Physical radius (m)': 'Simulation grain radius (m)',
    'Represented grain radius (m, 0 = same)': 'Real grain radius (m, 0 = simulation)',
    'Normal damping (Ns/m)': 'Restitution',
    'Liquid viscosity for drag (Pa s)':
        'removed: drag uses the liquid substance viscosity (one physical value, one home)',
}


def label_of(literal):
    text = json.loads(literal)
    visible = text.split('##')[0].strip()
    # A hidden-label widget ("##DomainBackend") is keyed by its id.
    return visible or ('##' + text.split('##', 1)[1].strip() if '##' in text else '')


def scan():
    widgets = []
    for pattern in FILES:
        for path in sorted(UI.glob(pattern)):
            text = path.read_text(encoding='utf-8', errors='replace')
            tabs = [(m.start(), m.group(1).strip()) for m in TAB.finditer(text)]
            for m in WIDGET.finditer(text):
                label = label_of(m.group(2))
                if not label:
                    continue
                tab = ''
                for start, name in tabs:
                    if start < m.start():
                        tab = name
                widgets.append({
                    'label': label,
                    'kind': m.group(1),
                    'target': ' '.join(m.group(3).split()).lstrip('&'),
                    'file': path.name,
                    'line': text.count('\n', 0, m.start()) + 1,
                    'tab': tab,
                })
    return widgets


# New helper files follow the standard: every value widget has a tooltip
# (DomainUi rows carry one by signature; a raw ImGui widget must be followed by
# DomainUi::tooltip or SetTooltip before the next widget).
TOOLTIP_FILES = ['Matter*Controls.cpp', 'DomainPanel*.cpp']
RAW = re.compile(r'ImGui::(Checkbox|Combo|Slider\w*|Drag\w*|Input\w*|ColorEdit\w*)\s*\(')


def missing_tooltips():
    missing = []
    for pattern in TOOLTIP_FILES:
        for path in sorted(UI.glob(pattern)):
            text = path.read_text(encoding='utf-8', errors='replace')
            matches = list(RAW.finditer(text))
            for n, m in enumerate(matches):
                end = matches[n+1].start() if n+1 < len(matches) else len(text)
                window = text[m.end():min(end, m.end()+1200)]
                if 'tooltip(' not in window and 'SetTooltip(' not in window:
                    missing.append(f"{path.name}:{text.count(chr(10), 0, m.start())+1}")
    return missing


# A standard-width field takes 55% of the row, so two on one line overflow the
# panel (Max Auto Resolution + Boundary Padding did, 2026-10-06).
SAMELINE_WIDE = re.compile(r'ImGui::SameLine\([^)]*\);\s*ImGui::SetNextItemWidth\(\s*DomainUi::itemWidth\(\)')


def wide_on_same_line():
    hits = []
    for pattern in FILES:
        for path in sorted(UI.glob(pattern)):
            text = path.read_text(encoding='utf-8', errors='replace')
            for m in SAMELINE_WIDE.finditer(text):
                hits.append(f"{path.name}:{text.count(chr(10), 0, m.start())+1}")
    return hits


def key(widget):
    return widget['label']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-baseline', action='store_true')
    args = parser.parse_args()
    current = scan()
    if args.write_baseline:
        BASELINE.write_text(json.dumps(current, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'baseline: {len(current)} widgets, {len({key(w) for w in current})} labels -> {BASELINE.name}')
        return 0
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    now = {}
    for w in current:
        now.setdefault(key(w), []).append(w)
    # Multiset: two widgets sharing a label (per-phase Viscosity) must both survive.
    used = {}
    lost, moved = [], []
    for w in baseline:
        k = key(w)
        change = CHANGES.get(k)
        if change and change.startswith('removed:'):
            continue
        target = change or k
        used[target] = used.get(target, 0) + 1
        if used[target] > len(now.get(target, [])):
            lost.append(w)
            continue
        tabs = {x['tab'] for x in now[target]}
        if w['tab'] not in tabs:
            moved.append((k, w['tab'], sorted(tabs)))
    for k, old, new in sorted(set((m[0], m[1], tuple(m[2])) for m in moved)):
        print(f'moved  {k!r}: {old or "-"} -> {", ".join(new) or "-"}')
    for w in lost:
        print(f"LOST   {w['label']!r} ({w['kind']} -> {w['target']}) was {w['file']}:{w['line']} tab {w['tab']!r}")
    stale = [k for k in CHANGES if k not in {key(w) for w in baseline}]
    for k in stale:
        print(f'STALE  CHANGES entry {k!r} names no baseline widget')
    untipped = missing_tooltips()
    for where in untipped:
        print(f'NO TIP {where}')
    wide = wide_on_same_line()
    for where in wide:
        print(f'WIDE   {where}: standard-width field after SameLine overflows the row')
    if lost or stale or untipped or wide:
        print(f'FAIL domain panel fields: {len(lost)} lost, {len(stale)} stale change entries, '
              f'{len(untipped)} widgets without tooltip, {len(wide)} wide fields on one line')
        return 1
    print(f'PASS domain panel fields: {len(baseline)} baseline widgets kept or changed on purpose '
          f'({len(moved)} moved, {len(current)} now)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
