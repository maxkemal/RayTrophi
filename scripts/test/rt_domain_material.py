"""Domain material for IPC test scripts (2026-10-07).

A liquid/matter domain's physics (viscosity, granular skeleton, freeze point,
conduction, chemistry) are its DEFAULT SUBSTANCE's now, resolved every step
(docs/dev/MADDE_TIPLERI_TASARIMI.md). fluid.set_param refuses the old keys.
A script that tuned them derives a project substance and points the domain
at it:

    from rt_domain_material import domain_material
    name = domain_material(call, 'T: stiff sand', 'Sand',
                           granular_young_modulus=2e6, granular_cohesion=0.)
    call('fluid.set_param', domain=DOMAIN, default_substance=name)

`call(method, **params)` is the script's own IPC wrapper.
"""

# Old fluid.set_param preset names -> built-in substances.
PRESET_SUBSTANCE = {
    'water': 'Water', 'oil': 'Oil', 'mud': 'Mud', 'honey': 'Honey',
    'chocolate': 'Chocolate', 'sand': 'Sand', 'gravel': 'Gravel', 'wax': 'Wax',
}

# Old domain keys -> substance fields (the same table fluid.set_param refuses with).
DOMAIN_TO_SUBSTANCE = {
    'kinematic_viscosity': 'liquid_kinematic_viscosity',
    'granular_friction_angle': 'granular_friction_degrees',
    'granular_cohesion': 'granular_cohesion',
    'granular_dilatancy': 'granular_dilatancy_degrees',
    'granular_young_modulus': 'granular_young_modulus',
    'granular_poisson_ratio': 'granular_poisson_ratio',
    'granular_tensile_cutoff': 'granular_tensile_cutoff',
    'granular_hardening': 'granular_hardening',
    'granular_fracture_strain': 'granular_fracture_strain',
    'granular_damage_rate': 'granular_damage_rate',
    'granular_healing_rate': 'granular_healing_rate',
    'granular_rebonding': 'granular_rebonding',
    'granular_softening_temperature': 'granular_softening_kelvin',
    'granular_softening_range': 'granular_softening_range',
    'granular_residual_strength': 'granular_residual_strength',
    'granular_tack_peak': 'granular_tack_peak',
    'granular_thermal_conductivity': 'parcel_conduction',
    'thermal_freeze_kelvin': 'melt_kelvin',
    'thermal_viscosity_range': 'liquid_freeze_viscosity_range',
    'thermal_cold_viscosity': 'liquid_cold_viscosity',
}


def substance_fields(**domain_keys):
    """Old domain keys -> a substance.set `fields` object."""
    fields = {}
    for key, value in domain_keys.items():
        if key == 'granular_enabled':
            fields['default_constitutive_model'] = 'granular' if value else 'fluid'
        elif key == 'thermal_freeze_kelvin':
            fields['meltable'] = True
            fields['melt_kelvin'] = value
        else:
            fields[DOMAIN_TO_SUBSTANCE[key]] = value
    return fields


def domain_material(call, name, based_on, **domain_keys):
    """Derive (or re-patch) project substance `name` from `based_on`; returns name.

    Re-running a script re-patches the same substance instead of failing on
    the existing name, so its values always match this call.
    """
    rows = {row['name']: row for row in call('substance.list')['substances']}
    if name not in rows:
        call('substance.derive', name=name, based_on=based_on)
    elif rows[name]['based_on'] != based_on:
        raise ValueError("substance {!r} already exists with a different base".format(name))
    fields = substance_fields(**domain_keys)
    if fields:
        call('substance.set', name=name, fields=fields)
    return name


def remove_material(call, name):
    """Best effort: refused while something still names it (that is the point)."""
    try:
        call('substance.remove', name=name)
        return True
    except Exception:  # noqa: BLE001 - the server's refusal is the information
        return False
