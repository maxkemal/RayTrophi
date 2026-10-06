"""Grain parameter conversions shared by the grain IPC tests.

The grain contact is authored by its coefficient of restitution e; the step
derives c = 2 zeta sqrt(k m_eff) per contact. Tests written against the old
per-domain normal damping c (N s/m) keep their physics through the same
conversion SceneSerializer applies to old scenes: grain-grain effective mass
m/2 of a 2667 kg/m^3 grain (sand bulk 1600 / packing .6).
"""
import math


def restitution_of(damping_n_s_m, stiffness_n_m, radius_m=.025, density=2667.):
    mass = density*4/3*math.pi*radius_m**3
    zeta = damping_n_s_m/(2*math.sqrt(stiffness_n_m*.5*mass))
    if zeta >= 1:
        return .01
    return min(1., max(.01, math.exp(-math.pi*zeta/math.sqrt(1-zeta*zeta))))
