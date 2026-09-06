"""Structural validator for a GLB/glTF written by GltfDirectWriter.

Checks the things a viewer would silently tolerate or silently mis-render:
every accessor's bytes are actually inside the buffer, index values are inside
their primitive's vertex range, JOINTS_0 indices exist in the skin, and the
required POSITION min/max are present.
"""
import json
import struct
import sys
import os

COMP = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2),
        5125: ('I', 4), 5126: ('f', 4)}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] == b'glTF':
        total = struct.unpack('<I', data[8:12])[0]
        assert total == len(data), f"GLB header length {total} != file size {len(data)}"
        off = 12
        js, binchunk = None, b''
        while off < len(data):
            clen, ctype = struct.unpack('<II', data[off:off + 8])
            body = data[off + 8: off + 8 + clen]
            if ctype == 0x4E4F534A:
                js = json.loads(body.decode('utf-8'))
            elif ctype == 0x004E4942:
                binchunk = body
            off += 8 + clen
            assert off % 4 == 0, f"chunk not 4-aligned at {off}"
        return js, binchunk
    js = json.loads(data.decode('utf-8'))
    buf = b''
    if js.get('buffers'):
        uri = js['buffers'][0].get('uri')
        if uri and not uri.startswith('data:'):
            with open(os.path.join(os.path.dirname(path), uri), 'rb') as f:
                buf = f.read()
    return js, buf


def acc_read(g, buf, i):
    a = g['accessors'][i]
    bv = g['bufferViews'][a['bufferView']]
    fmt, size = COMP[a['componentType']]
    n = NCOMP[a['type']]
    start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    count = a['count']
    need = count * n * size
    assert start + need <= len(buf), (
        f"accessor {i} reads {start}+{need} = {start+need} but buffer is {len(buf)}")
    raw = buf[start:start + need]
    return struct.unpack('<' + fmt * (count * n), raw), n


def check(path):
    g, buf = load(path)
    errs, warns, notes = [], [], []

    declared = g.get('buffers', [{}])[0].get('byteLength', 0) if g.get('buffers') else 0
    if declared and declared != len(buf):
        errs.append(f"buffers[0].byteLength={declared} but chunk holds {len(buf)}")

    for key in ('materials', 'accessors', 'bufferViews', 'meshes', 'nodes'):
        if key in g and not g[key]:
            errs.append(f'"{key}" present but empty (invalid glTF)')

    # bufferViews inside the buffer
    for i, bv in enumerate(g.get('bufferViews', [])):
        end = bv.get('byteOffset', 0) + bv['byteLength']
        if end > len(buf):
            errs.append(f"bufferView {i} ends at {end}, buffer is {len(buf)}")

    tri = 0
    vert = 0
    for mi, m in enumerate(g.get('meshes', [])):
        for pi, p in enumerate(m['primitives']):
            attrs = p['attributes']
            pos_i = attrs['POSITION']
            pos = g['accessors'][pos_i]
            if 'min' not in pos or 'max' not in pos:
                errs.append(f"mesh{mi}/prim{pi}: POSITION accessor has no min/max (required)")
            vcount = pos['count']
            if pi == 0:
                vert += vcount
            vals, n = acc_read(g, buf, pos_i)
            if any(v != v for v in vals):  # NaN
                errs.append(f"mesh{mi}/prim{pi}: POSITION contains NaN")
            mn = [min(vals[k::3]) for k in range(3)]
            mx = [max(vals[k::3]) for k in range(3)]
            for k in range(3):
                if abs(mn[k] - pos['min'][k]) > 1e-3 or abs(mx[k] - pos['max'][k]) > 1e-3:
                    errs.append(f"mesh{mi}/prim{pi}: POSITION min/max wrong on axis {k}: "
                                f"declared {pos['min'][k]}..{pos['max'][k]} actual {mn[k]}..{mx[k]}")
            for name in ('NORMAL', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'):
                if name in attrs:
                    a = g['accessors'][attrs[name]]
                    if a['count'] != vcount:
                        errs.append(f"mesh{mi}/prim{pi}: {name} count {a['count']} != POSITION {vcount}")
                    acc_read(g, buf, attrs[name])
            if 'indices' in p:
                idx, _ = acc_read(g, buf, p['indices'])
                if len(idx) % 3:
                    errs.append(f"mesh{mi}/prim{pi}: index count {len(idx)} not a multiple of 3")
                tri += len(idx) // 3
                bad = max(idx) if idx else -1
                if bad >= vcount:
                    errs.append(f"mesh{mi}/prim{pi}: index {bad} out of range (vcount {vcount})")
            if 'material' in p and p['material'] >= len(g.get('materials', [])):
                errs.append(f"mesh{mi}/prim{pi}: material {p['material']} out of range")

    # skins
    for si, s in enumerate(g.get('skins', [])):
        joints = s['joints']
        for j in joints:
            if j >= len(g['nodes']):
                errs.append(f"skin{si}: joint node {j} out of range")
        if 'inverseBindMatrices' in s:
            ibm = g['accessors'][s['inverseBindMatrices']]
            if ibm['count'] != len(joints):
                errs.append(f"skin{si}: inverseBindMatrices count {ibm['count']} != joints {len(joints)}")
            acc_read(g, buf, s['inverseBindMatrices'])
        else:
            warns.append(f"skin{si}: no inverseBindMatrices")

    # every JOINTS_0 index must exist in the skin used by the node
    node_skin = {}
    for ni, n in enumerate(g.get('nodes', [])):
        if 'skin' in n and 'mesh' in n:
            node_skin[n['mesh']] = n['skin']
    for mi, m in enumerate(g.get('meshes', [])):
        if mi not in node_skin:
            continue
        njoints = len(g['skins'][node_skin[mi]]['joints'])
        for pi, p in enumerate(m['primitives']):
            if 'JOINTS_0' not in p['attributes']:
                errs.append(f"mesh{mi}/prim{pi}: node has a skin but primitive has no JOINTS_0")
                continue
            jv, _ = acc_read(g, buf, p['attributes']['JOINTS_0'])
            if jv and max(jv) >= njoints:
                errs.append(f"mesh{mi}/prim{pi}: JOINTS_0 index {max(jv)} >= joint count {njoints}")
            wv, _ = acc_read(g, buf, p['attributes']['WEIGHTS_0'])
            offenders = 0
            for k in range(0, len(wv), 4):
                s4 = wv[k] + wv[k+1] + wv[k+2] + wv[k+3]
                if abs(s4 - 1.0) > 1e-3 and s4 != 0.0:
                    offenders += 1
            if offenders:
                errs.append(f"mesh{mi}/prim{pi}: {offenders} vertices have WEIGHTS_0 not summing to 1")

    # animations
    for ai, a in enumerate(g.get('animations', [])):
        for ci, c in enumerate(a['channels']):
            t = c['target']
            if 'node' in t and t['node'] >= len(g['nodes']):
                errs.append(f"animation{ai}/channel{ci}: target node out of range")
            s = a['samplers'][c['sampler']]
            inp = g['accessors'][s['input']]
            out = g['accessors'][s['output']]
            if 'min' not in inp or 'max' not in inp:
                errs.append(f"animation{ai}/channel{ci}: input accessor lacks min/max (required)")
            if inp['count'] != out['count']:
                errs.append(f"animation{ai}/channel{ci}: input count {inp['count']} != output {out['count']}")
            expect = {'translation': 'VEC3', 'rotation': 'VEC4', 'scale': 'VEC3', 'weights': 'SCALAR'}
            if out['type'] != expect.get(t['path']):
                errs.append(f"animation{ai}/channel{ci}: path {t['path']} wants "
                            f"{expect.get(t['path'])}, output is {out['type']}")
            acc_read(g, buf, s['input'])
            acc_read(g, buf, s['output'])

    # images
    for ii, im in enumerate(g.get('images', [])):
        if 'bufferView' in im:
            bv = g['bufferViews'][im['bufferView']]
            blob = buf[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']]
            if im.get('mimeType') == 'image/png' and blob[:8] != b'\x89PNG\r\n\x1a\n':
                errs.append(f"image{ii}: declared PNG but magic is {blob[:8]!r}")
            if im.get('mimeType') == 'image/jpeg' and blob[:2] != b'\xff\xd8':
                errs.append(f"image{ii}: declared JPEG but magic is {blob[:2]!r}")

    inst_nodes = [n for n in g.get('nodes', [])
                  if 'EXT_mesh_gpu_instancing' in n.get('extensions', {})]
    inst_total = 0
    for n in inst_nodes:
        at = n['extensions']['EXT_mesh_gpu_instancing']['attributes']
        counts = {k: g['accessors'][v]['count'] for k, v in at.items()}
        if len(set(counts.values())) != 1:
            errs.append(f"instancing node '{n.get('name')}': attribute counts disagree {counts}")
        inst_total += list(counts.values())[0]
        for v in at.values():
            acc_read(g, buf, v)

    for ext in g.get('extensionsUsed', []):
        notes.append(f"extensionsUsed: {ext}")

    print(f"--- {os.path.basename(path)} ({len(open(path,'rb').read())/1048576:.2f} MB)")
    print(f"    meshes={len(g.get('meshes',[]))} prims={sum(len(m['primitives']) for m in g.get('meshes',[]))} "
          f"tris={tri} verts={vert} nodes={len(g.get('nodes',[]))} mats={len(g.get('materials',[]))} "
          f"imgs={len(g.get('images',[]))} skins={len(g.get('skins',[]))} anims={len(g.get('animations',[]))} "
          f"cams={len(g.get('cameras',[]))} instNodes={len(inst_nodes)} instances={inst_total}")
    if g.get('animations'):
        for a in g['animations']:
            paths = sorted({c['target']['path'] for c in a['channels']})
            print(f"    anim '{a.get('name')}': {len(a['channels'])} channels {paths}")
    for n in notes:
        print(f"    {n}")
    for w in warns:
        print(f"    WARN  {w}")
    for e in errs:
        print(f"    ERROR {e}")
    if not errs:
        print("    OK - structurally valid")
    return len(errs)


if __name__ == '__main__':
    bad = 0
    for p in sys.argv[1:]:
        try:
            bad += check(p)
        except Exception as ex:
            print(f"--- {p}\n    ERROR {type(ex).__name__}: {ex}")
            bad += 1
        print()
    sys.exit(1 if bad else 0)
