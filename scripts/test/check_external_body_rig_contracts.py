"""Read-only C2 source/project wiring check. No C++ build or application launch."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def main():
    ns = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
    shared = ("Input/ExternalBodyRigMath.cpp", "Input/ExternalBodyRig.cpp",
              "Api/RtApiExternalBody.cpp")
    for target in ("RayTrophiStudio", "RayTrophiPlayer"):
        xml = ET.parse(ROOT/f"RayTrophiStudio/{target}.vcxproj")
        sources = [item.attrib["Include"].replace("\\", "/")
                   for item in xml.findall(".//m:ClCompile", ns) if "Include" in item.attrib]
        for path in shared:
            assert sources.count("source/src/" + path) == 1, (target, path)
        assert sources.count("source/src/Api/RtPythonExternalBody.cpp") == (target == "RayTrophiStudio")
    descriptor = (SOURCE/"src/Api/RtIpcMethodDescriptors.cpp").read_text()
    for name in ("anim_set_external_pose", "anim_external_poses", "anim_clear_external_pose"):
        assert "desc_" + name + " =" in descriptor, name
    params = descriptor.split("params_anim_set_external_pose[] = {", 1)[1].split("};", 1)[0]
    for field in ('{"character", "string", true', '{"person_id", "int", true',
                  '{"config", "object", true'):
        assert field in params, field
    rig_math = (SOURCE/"src/Input/ExternalBodyRigMath.cpp").read_text()
    for operation in ("solveIKPose", "constrainJointPose", "matchIKPose"):
        assert "RigAuthoring::" + operation in rig_math
    kinematic = (SOURCE/"src/Physics/KinematicColliderScene.cpp").read_text()
    assert "externalBodyRigColliderAvailable" in kinematic
    assert "externalBodyRigColliderBoneAvailable" in kinematic
    body_api = (SOURCE/"src/Api/RtApiBodyPose.cpp").read_text()
    assert "tickExternalBodyRigs(g_ctx->scene)" in body_api
    clock = (SOURCE/"src/Game/GameRuntime.cpp").read_text()
    assert clock.index("tickExternalBodyRigs(scene)") < clock.index("if (!session().configured())")
    for path in (SOURCE/"src/Input").glob("ExternalBodyRig*.cpp"):
        code = path.read_text()
        assert "JoltWorld.h" not in code and "createBody(" not in code
        assert "dynamic_pointer_cast<Triangle>" not in code
        assert "downloadBuffer" not in code
    print("PASS: shared IK/limits, flat path, existing proxy gates, IPC schemas and target registration")


if __name__ == "__main__":
    main()
