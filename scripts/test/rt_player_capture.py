"""Temporary diagnostic screenshot of an already open Player; restores capture."""
import base64
import time
from pathlib import Path
from rt_ipc import RtIpc, PLAYER_PIPE_NAME

ipc = RtIpc(PLAYER_PIPE_NAME)
enabled = ipc.call("viewport.status")["capture_enabled"]
try:
    ipc.call("viewport.capture", enabled=True)
    time.sleep(0.3)
    result = ipc.call("viewport.get_screenshot")
    data = result["image_base64"]
    path = Path("docs/dev/.captures/player-ghost.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data.split(",")[-1]))
    print(path.resolve())
finally:
    ipc.call("viewport.capture", enabled=enabled)
