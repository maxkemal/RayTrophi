"""Evaluate actual CPU/GLSL scalar rebound expressions without compiling either."""
from pathlib import Path
import ast
import math
import re

ROOT = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source"


def expression(path, declaration):
    text = (ROOT / path).read_text(encoding="utf-8")
    value = re.search(declaration + r" = ([^;]+);", text).group(1)
    value = value.replace("wall.restitution", "restitution").replace("std::max", "max")
    value = re.sub(r"(?<=\d)f\b", "", value)
    value = value.replace("wall.adhesion_pa", "adhesion").replace("wall.mass_per_area", "areal_mass")
    value = " ".join(value.split())
    tree = ast.parse(value, mode="eval")
    for node in ast.walk(tree):
        assert isinstance(node, (ast.Expression, ast.BinOp, ast.Mult, ast.Sub,
            ast.Add, ast.Div, ast.Call, ast.Name, ast.Load, ast.Constant)), value
        if isinstance(node, ast.Name):
            assert node.id in {"max", "restitution", "incoming", "support", "rebound",
                               "glue", "vn", "adhesion", "dt", "areal_mass"}
    return compile(tree, path, "eval")


def main():
    for path, prefix in (("include/Fluid/DomainWallContact.h", "const float"),
                         ("shaders/sim_domain_wall_contact.glsl", "float")):
        rebound = expression(path, prefix + " rebound")
        normal_change = expression(path, prefix + " normal_change")
        for dt in (1 / 24, 1 / 60, 1 / 480):
            for restitution in (0.0, 0.4, 1.0):
                support = 9.81 * dt
                for impact_speed in (0.0, 0.1, 4.0):
                    values = dict(restitution=restitution, support=support,
                                  incoming=impact_speed + support, max=max)
                    values["rebound"] = eval(rebound, {"__builtins__": {}}, values)
                    assert math.isclose(values["rebound"], restitution * impact_speed,
                                        abs_tol=1e-12), (path, values)
                    correction = eval(normal_change, {"__builtins__": {}}, values)
                    assert correction >= values["incoming"]
        # Repeated resting support cannot generate a departing normal velocity.
        for dt in (1 / 24, 1 / 60, 1 / 480):
            speed = 0.0
            for _ in range(1000):
                values = dict(restitution=0.4, support=9.81 * dt,
                              incoming=9.81 * dt - speed, max=max)
                speed = eval(rebound, {"__builtins__": {}}, values)
                assert speed == 0.0, (path, speed)
    # Evaluate source expressions against analytic impulse/cohesion oracles.
    cpu_glue = expression("include/Fluid/DomainWallContact.h", "const float glue")
    for dt in (0.001, 0.01, 0.1):
        values = dict(adhesion=10000.0, dt=dt, areal_mass=1000.0, max=max)
        glue = eval(cpu_glue, {"__builtins__": {}}, values)
        assert math.isclose(glue, 10.0 * dt, abs_tol=1e-12)
        for path, prefix in (("include/Fluid/DomainWallContact.h", "const float"),
                             ("shaders/sim_domain_wall_contact.glsl", "float")):
            cohesion = expression(path, prefix + " cohesion")
            for separation, expected in ((0.0, glue), (glue / 2, glue / 2),
                                         (glue * 2, 0.0)):
                actual = eval(cohesion, {"__builtins__": {}},
                              dict(glue=glue, vn=separation, max=max))
                assert math.isclose(actual, expected, abs_tol=1e-12), (path, actual)
    print("PASS CPU/GLSL adhesion impulse and finite cohesive budget")
    print("PASS CPU/GLSL actual rebound expressions: resting support stays still; impacts rebound")


if __name__ == "__main__":
    main()
