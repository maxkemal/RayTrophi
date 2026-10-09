"""Grain material for IPC test scripts (2026-10-09, MADDE_UI_TEK_OTORITE U2/U3).

The DEM grain material (friction, rolling, restitution, packing, water capacity,
...) is the grain SUBSTANCE's now, not the domain's; fluid.set_grain_settings
refuses those keys and names their new home. `enabled` and `wet_grains` are
derived; the contact stiffness is `stiffness_scale` x 8e5 N/m per m x radius.

Scripts written against the old keys keep working with one line after the
client is created:

    from rt_grain_material import install_grain_material
    client = RtIpc()
    install_grain_material(client)

The installed translator, per domain:
  * sends the domain keys (radius, numerics, coupling, sleep) as before, with
    stiffness_n_m converted to stiffness_scale for the domain's radius;
  * puts the material keys on a derived test substance
    "T: <domain> <base> grains" (based on the source's Sand/Gravel/Ice);
  * points every flow source aimed at that domain at the derived substance and
    carries birth_saturation as the source's grain_birth_saturation;
  * drops `enabled` (grains follow the substance).
"""

STIFFNESS_PER_RADIUS = 8.0e5
DEM_BASES = {'Sand', 'Gravel', 'Ice'}

# Old domain key -> substance field.
MATERIAL = {
    'friction': 'grain_friction',
    'rolling_friction': 'grain_rolling_friction',
    'twisting_friction': 'grain_twisting_friction',
    'restitution': 'grain_restitution',
    'tangential_stiffness_ratio': 'grain_tangential_stiffness_ratio',
    'packing_fraction': 'grain_packing_fraction',
    'represented_grain_radius_m': 'grain_real_radius_m',
    'water_capacity_fraction': 'grain_water_capacity_fraction',
    'absorption_rate_per_s': 'grain_absorption_rate_per_s',
    'drying_rate_per_s': 'grain_drying_rate_per_s',
    'contact_angle_deg': 'grain_contact_angle_deg',
}
# wet_grains on with no explicit capacity: the old default capacity.
DEFAULT_WET_CAPACITY = 0.05


def derived_name(domain, base):
    return 'T: {} {} grains'.format(domain, base)


class _GrainTranslator:
    def __init__(self, raw_call):
        self.raw = raw_call
        self.domains = {}   # domain -> {'radius', 'fields', 'birth', 'derived': {base: name}}
        self.sources = {}   # source -> {'domain', 'base'}
        self.applied = {}   # derived substance -> fields already written

    def _state(self, domain):
        if domain not in self.domains:
            radius = .025
            try:
                radius = float(self.raw('fluid.grain_settings', domain=domain)['radius_m'])
            except Exception:  # noqa: BLE001 - a missing domain keeps the default
                pass
            self.domains[domain] = {'radius': radius, 'fields': {}, 'birth': None, 'derived': {}}
        return self.domains[domain]

    def _ensure(self, domain, base):
        state = self._state(domain)
        name = state['derived'].get(base)
        if name is None:
            name = derived_name(domain, base)
            rows = {row['name']: row for row in self.raw('substance.list')['substances']}
            if name not in rows:
                self.raw('substance.derive', name=name, based_on=base)
            state['derived'][base] = name
        # Send only what changed: a re-sent grain_packing_fraction is refused
        # while the domain holds live grains (fixed at birth).
        applied = self.applied.setdefault(name, {})
        changed = {k: v for k, v in state['fields'].items() if applied.get(k) != v}
        if changed:
            self.raw('substance.set', name=name, fields=changed)
            applied.update(changed)
        return name

    def grain_settings(self, params):
        params = dict(params)
        domain = params['domain']
        state = self._state(domain)
        params.pop('enabled', None)
        if 'radius_m' in params:
            state['radius'] = float(params['radius_m'])
        if 'stiffness_n_m' in params:
            params['stiffness_scale'] = float(params.pop('stiffness_n_m')) / (
                STIFFNESS_PER_RADIUS * state['radius'])
        material = {}
        for old, new in MATERIAL.items():
            if old in params:
                material[new] = params.pop(old)
        if 'wet_grains' in params:
            wet = params.pop('wet_grains')
            if not wet:
                material['grain_water_capacity_fraction'] = 0.0
            elif 'grain_water_capacity_fraction' not in material and \
                    not state['fields'].get('grain_water_capacity_fraction'):
                material['grain_water_capacity_fraction'] = DEFAULT_WET_CAPACITY
        if 'birth_saturation' in params:
            state['birth'] = float(params.pop('birth_saturation'))
        if 'surface_tension_n_m' in params:
            raise ValueError('surface_tension_n_m is the liquid substance\'s '
                             '(liquid_surface_tension_n_m); derive the liquid instead')
        if material:
            state['fields'].update(material)
            for base in list(state['derived']):
                self._ensure(domain, base)
            # Sources created before the material: point them at it now.
            for source, info in self.sources.items():
                if info['domain'] == domain and info['base'] in DEM_BASES:
                    self.raw('flow_source.update', name=source,
                             fluid_substance=self._ensure(domain, info['base']))
        result = self.raw('fluid.set_grain_settings', **params)
        if state['birth'] is not None:
            for source, info in self.sources.items():
                if info['domain'] == domain:
                    self.raw('flow_source.update', name=source,
                             grain_birth_saturation=state['birth'])
        return result

    def flow_source(self, method, params):
        params = dict(params)
        name = params.get('name')
        known = self.sources.get(name, {})
        domain = params.get('domain', known.get('domain'))
        base = params.get('fluid_substance', known.get('base'))
        if base is not None and base.startswith('T: ') and base.endswith(' grains'):
            base = known.get('base', base)
        if name is not None:
            self.sources[name] = {'domain': domain, 'base': base}
        state = self.domains.get(domain) if domain is not None else None
        if state is not None and base in DEM_BASES and \
                (state['fields'] or state['birth'] is not None):
            params['fluid_substance'] = self._ensure(domain, base)
            if state['birth'] is not None:
                params['grain_birth_saturation'] = state['birth']
        return self.raw(method, **params)

    def __call__(self, method, **params):
        if method == 'fluid.set_grain_settings':
            return self.grain_settings(params)
        if method in ('flow_source.create', 'flow_source.update'):
            return self.flow_source(method, params)
        return self.raw(method, **params)


def install_grain_material(client):
    """Route `client.call` through the grain-material translator; returns client."""
    translator = _GrainTranslator(client.call)
    client.call = translator
    return client
