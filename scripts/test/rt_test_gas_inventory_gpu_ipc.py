"""External IPC acceptance after one matching C++/simulation shader build.

Run in a paused disposable scene. Reuses boiling and heat conservation fixtures,
which reset simulation and restore authored switches, not the original live state.
Does not launch the app. No invented performance threshold.
"""
import json
import math
from pathlib import Path

from rt_ipc import RtIpc
import rt_test_boiling_ipc as boiling
import rt_test_heat_exchange_ipc as heat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/gas_inventory_gpu_acceptance_live.json'


def main():
    rows = []
    original_call = RtIpc.call
    current = [None]

    def call(self, method, /, **params):
        result = original_call(self, method, **params)
        if method == 'fluid.step':
            stats = original_call(self, 'gas.step_stats', domain=current[0])
            assert not stats.get('inventory_step_held'), stats
            if stats.get('measured'):
                domain = original_call(self, 'fluid.get', domain=current[0])
                mass, energy = domain['gas_phase_mass_kg'], domain['gas_phase_energy_j']
                assert math.isfinite(mass) and math.isfinite(energy) and mass >= 0 and energy >= 0
                ledger = original_call(self, 'matter.exchanges')
                deposited = sum(r['target_mass_kg'] for r in ledger['exchanges']
                                if r['target'] == current[0] and r['kind'] == 'vaporization')
                # Vapour may turn to fog water or condense back onto the liquid
                # within the step (2026-10-11): count the air's water and the return.
                returned = sum(r['target_mass_kg'] for r in ledger['exchanges']
                               if (r['source'] == current[0] and r['kind'] == 'condensation') or
                               r['kind'] == 'precipitation')
                # A non-water boil goes to the non-condensable pool (part 4).
                in_air = (mass + domain.get('gas_condensate_kg', 0.0) +
                          domain.get('gas_rain_kg', 0.0) + domain.get('gas_foreign_kg', 0.0))
                assert in_air + 1e-8 >= .95 * (deposited - returned), (in_air, deposited, ledger)

                assert stats.get('inventory_transport') in ('gpu_resident', 'gpu_resident_idle'), stats
                assert stats.get('inventory_grid_upload_bytes') == 0, stats
                assert stats.get('inventory_grid_download_bytes') == 0, stats
                first = original_call(self, 'gas.step_stats', domain=current[0])
                original_call(self, 'fluid.get', domain=current[0])
                second = original_call(self, 'gas.step_stats', domain=current[0])
                assert first['inventory_query_download_bytes'] == second['inventory_query_download_bytes'], second
                rows.append({'domain': current[0], 'transport': stats['inventory_transport'],
                             'inventory_ms': stats['inventory_advection_ms'],
                             'surface_ms': stats['gpu_surface_exchange_ms'],
                             'gas_total_ms': stats['total_ms'], 'mass_kg': mass, 'energy_j': energy,
                             'grid_upload_bytes': stats['inventory_grid_upload_bytes'],
                             'grid_download_bytes': stats['inventory_grid_download_bytes'],
                             'snapshot_bytes': second['inventory_snapshot_download_bytes'],
                             'query_bytes': second['inventory_query_download_bytes']})
        return result

    RtIpc.call = call
    try:
        for fixture in (boiling, heat):
            current[0] = fixture.DOMAIN
            fixture.main()
            assert any(r['domain'] == fixture.DOMAIN and r['mass_kg'] > 0 for r in rows), fixture.DOMAIN
        print('PASS resident GPU inventory and reused boiling/heat conservation fixtures;', len(rows), 'samples')
    finally:
        RtIpc.call = original_call
        OUT.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
