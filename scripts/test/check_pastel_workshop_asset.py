"""Read the exported binary independently; verify playable asset/UV invariants."""
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "RayTrophiStudio/assets/scenes/pastel-paint-workshop/workshop.rtp"


def read_flat(path):
    with path.open("rb") as stream:
        def uint():
            return struct.unpack("<I", stream.read(4))[0]
        assert stream.read(4) == b"RTP8" and uint() == 8
        transforms = [np.frombuffer(stream.read(100), dtype="<f4").copy()
                      for _ in range(uint())]
        meshes = {}
        for _ in range(uint()):
            count = struct.unpack("<H",stream.read(2))[0]
            name = stream.read(count).decode()
            transform, vertices, indices = uint(), uint(), uint()
            p = np.frombuffer(stream.read(vertices*12),dtype="<f4").reshape(-1,3)
            n = np.frombuffer(stream.read(vertices*12),dtype="<f4").reshape(-1,3)
            uv = np.frombuffer(stream.read(vertices*8),dtype="<f4").reshape(-1,2)
            materials = np.frombuffer(stream.read(vertices*2),dtype="<u2")
            faces = np.frombuffer(stream.read(indices*4),dtype="<u4").reshape(-1,3)
            assert uint() == 0 and stream.read(1) == b"\0"
            assert name not in meshes
            meshes[name] = dict(p=p,n=n,uv=uv,materials=materials,faces=faces,
                                transform=transforms[transform][:16].reshape(4,4))
        assert uint() == 0 and not stream.read(1), "no facades or trailing payload expected"
        return meshes


def main():
    project = json.loads(ASSET.read_text())
    config = json.loads(Path(str(ASSET)+".balloons.json").read_text())
    shared = json.loads(Path(str(ASSET)+".shared").read_text())
    meshes = read_flat(Path(str(ASSET)+".bin"))
    materials = shared["materials"]["materials"]
    assert [m["id"] for m in materials] == list(range(len(materials)))
    total = 0
    for mesh in meshes.values():
        p,n,faces = mesh["p"],mesh["n"],mesh["faces"]
        assert np.isfinite(p).all() and np.isfinite(n).all()
        assert np.max(np.abs(np.linalg.norm(n,axis=1)-1)) < 1e-5
        assert faces.max() < len(p) and mesh["materials"].max() < len(materials)
        xyz = p[faces]
        area = np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        assert np.min(np.linalg.norm(area,axis=1)) > 1e-8, "degenerate triangle"
        assert np.all(np.einsum("ij,ij->i",area,n[faces].mean(axis=1)) > 0), "winding"
        assert np.isfinite(mesh["uv"]).all()
        total += len(faces)
    assert total == 17200 and len(meshes) == 32
    assert len(config["surfaces"]) == 4 and config["resolution"] == 512
    for name in config["surfaces"]:
        mesh = meshes[name]
        assert len(mesh["p"]) == 4 and len(mesh["faces"]) == 2
        uv = mesh["uv"][mesh["faces"]]
        area = np.abs(np.cross(uv[:,1]-uv[:,0],uv[:,2]-uv[:,0])) / 2
        assert np.isclose(area.sum(),1) and np.allclose(sorted(area),[0.5,0.5])
        assert np.allclose(mesh["uv"].min(axis=0),[0,0])
        assert np.allclose(mesh["uv"].max(axis=0),[1,1])
        material = mesh["materials"][0]
        assert all(not np.any(other["materials"]==material)
                   for other_name,other in meshes.items() if other_name != name)
    assert len(set(config["visual_nodes"])) == config["balloons"] == 12
    ids = []
    for name in config["visual_nodes"]:
        mesh = meshes[name]
        assert np.allclose(np.linalg.norm(mesh["p"],axis=1),0.325)
        assert len(mesh["faces"]) == 1216
        material = int(mesh["materials"][0])
        ids.append(material)
        assert materials[material]["clearcoat"] == 0.5
        roughness = materials[material]["roughnessProperty"]
        assert np.allclose(np.array(roughness["color"])*roughness["intensity"],0.18)
        assert materials[material]["transmission"] == 0
    assert len(set(ids)) == 12
    assert not config["autostart"]
    # Lighting is authored by the user; validate values, not the initial rig's count.
    for light in project["lights"]:
        assert light["type"] in (0, 1, 2, 3)
        assert np.isfinite(light["position"]).all()
        assert np.isfinite(light["color"]).all()
        assert np.isfinite(light["intensity"]) and light["intensity"] >= 0
    for system in project["particle_simulation"]["systems"]:
        assert len(system["colliders"]) == 4 and not system["domains"]
        assert [c["source_mode"] for c in system["colliders"]] == ["ObjectAABB"]*3+["PlaneY"]
        assert all(c["source_name"] in meshes for c in system["colliders"][:3])
    print("PASS: 32 flat meshes / 17,200 triangles; four independent full UV canvases; "
          "12 smooth PBR spheres; four simple colliders; no imported dependencies")


if __name__ == "__main__":
    main()
