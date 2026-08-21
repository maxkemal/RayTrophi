import sys
import os
sys.path.append(r"E:\RayTrophi_projesi\raytracing_Proje_Moduler\RayTrophiAgent")
from core.ipc_client import IPCClient

client = IPCClient()
if not client.connect(timeout_sec=5):
    print("Failed to connect to RayTrophi IPC!")
    sys.exit(1)

# List terrains
res = client.call("terrain.list", {})
print("Terrain list:", res)

terrains = res.get("result", [])
if not terrains:
    print("No terrains found! Will try to create one...")
    res = client.call("terrain.create", {"name": "Terrain"})
    print("Create terrain result:", res)
    res = client.call("terrain.list", {})
    terrains = res.get("result", [])

if not terrains:
    print("Still no terrain, exiting.")
    sys.exit(1)

tname = terrains[0]["name"]
print(f"Testing on terrain: {tname}")

print(">>> Testing terrain.set_paint_resolution to 2048")
res1 = client.call("terrain.set_paint_resolution", {"name": tname, "paint_resolution": 2048})
print("Result:", res1)

print(">>> Testing terrain.set_mesh_resolution to 512")
res2 = client.call("terrain.set_mesh_resolution", {"name": tname, "mesh_resolution": 512})
print("Result:", res2)

client.close()
