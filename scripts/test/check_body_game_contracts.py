"""Read-only source/registration audit. Does not compile or launch anything."""
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.perception_bridge.fake_children import load_plan
from tools.perception_bridge.schema import JOINTS, EVENTS


def main():
    source = ROOT / "RayTrophiStudio/source"
    core = (source / "include/Input/BodyPoseRuntime.h").read_text()
    joint_block = core.split("kBodyJoints = {", 1)[1].split("};", 1)[0]
    assert set(re.findall(r'"([a-z_]+)"', joint_block)) == set(JOINTS)
    rules = (source / "src/Game/GameRules.cpp").read_text()
    for event in EVENTS:
        assert f'"{event}"' in rules, event
    ns = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
    required = {
        "Input/BodyPoseRuntime.cpp", "Game/GameRules.cpp", "Game/GameSession.cpp",
        "Game/GameRuntime.cpp", "Api/RtApiBodyPose.cpp", "Api/RtIpcBodyPose.cpp",
        "Api/RtApiGame.cpp", "Api/RtIpcGame.cpp"
    }
    for project in ("RayTrophiStudio", "RayTrophiPlayer"):
        tree = ET.parse(ROOT / f"RayTrophiStudio/{project}.vcxproj")
        rows = [item.attrib["Include"].replace("\\", "/")
                for item in tree.findall(".//m:ClCompile", ns) if "Include" in item.attrib]
        for path in required:
            assert rows.count("source/src/" + path) == 1, (project, path)
        for path in ("Api/RtPythonBodyPose.cpp", "Api/RtPythonGame.cpp"):
            assert rows.count("source/src/" + path) == (project == "RayTrophiStudio")
    descriptors = (source / "src/Api/RtIpcMethodDescriptors.cpp").read_text()
    for method in ("input_set_body_poses", "input_body_poses", "input_clear_body_poses",
                   "game_session", "game_rules", "game_post_event"):
        assert f"desc_{method} =" in descriptors, method
    params = descriptors.split("params_input_set_body_poses[] = {", 1)[1].split("};", 1)[0]
    assert re.search(r'\{"t", "float", true', params)
    assert re.search(r'\{"persons", "array", true', params)
    plan = load_plan(ROOT / "tools/perception_bridge/session_plan.json")
    assert plan["expected_states"] == ["attract", "playing", "finale", "resetting", "attract"]
    assert set(plan["required_events"]).issubset(EVENTS)
    print("PASS: joint/event contract, both project registrations, descriptor schemas, session plan")


if __name__ == "__main__":
    main()
