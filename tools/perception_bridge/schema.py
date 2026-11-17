"""The 2026-11-15 body-pose contract; no camera or optional dependencies."""
import math

JOINTS = ("head", "neck", "pelvis") + tuple(
    f"{side}_{joint}" for side in ("l", "r")
    for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
EVENTS = ("gesture.stomp", "gesture.arms_up", "gesture.reach", "contact",
          "zone.enter", "zone.exit", "balloon.pop", "coverage.threshold",
          "timer", "session.state")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_pose(message, previous_t=None):
    """Raise ValueError on invalid packets; require the bridge's full 15-joint set."""
    if not isinstance(message, dict) or set(message) != {"t", "persons"}:
        raise ValueError("pose requires exactly t and persons")
    t = message["t"]
    if not number(t) or t < 0 or (previous_t is not None and t <= previous_t):
        raise ValueError("t must be finite, nonnegative and strictly increasing")
    persons = message["persons"]
    if not isinstance(persons, list) or len(persons) > 10:
        raise ValueError("persons must be a list of 0..10 people")
    seen = set()
    for person in persons:
        if not isinstance(person, dict) or set(person) != {"id", "confidence", "joints"}:
            raise ValueError("person requires id, confidence, joints")
        pid = person["id"]
        if type(pid) is not int or pid < 0 or pid > 2147483647 or pid in seen:
            raise ValueError("id must be a unique nonnegative int32")
        seen.add(pid)
        confidence = person["confidence"]
        if not number(confidence) or not 0 <= confidence <= 1:
            raise ValueError("person confidence must be in [0,1]")
        joints = person["joints"]
        if not isinstance(joints, dict) or set(joints) != set(JOINTS):
            raise ValueError("exactly the 15 canonical joints are required")
        for joint in joints.values():
            if not isinstance(joint, list) or len(joint) != 4:
                raise ValueError("joint requires [x,y,z,confidence]")
            if not all(number(v) for v in joint) or not 0 <= joint[3] <= 1:
                raise ValueError("nonfinite joint or confidence outside [0,1]")
            if any(abs(v) > 10000 for v in joint[:3]):
                raise ValueError("joint coordinate exceeds 10000 metres")
    return message
