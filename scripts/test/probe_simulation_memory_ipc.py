"""Read-only live allocation census. Does not step, reset, save or clear caches."""
import json
from collections import defaultdict

from rt_ipc import RtIpc


def main():
    client = RtIpc()
    try:
        report = client.call('perf.get_gpu_memory')
        simulation = report.get('simulation')
        if not simulation or not simulation.get('supported'):
            raise RuntimeError('Live compute allocation census unavailable; rebuild the C++ app')
        groups = defaultdict(lambda: {'count': 0, 'allocation_bytes': 0, 'requested_bytes': 0})
        for buffer in simulation['buffers']:
            group = groups[buffer['name']]
            group['count'] += 1
            group['allocation_bytes'] += buffer['allocation_bytes']
            group['requested_bytes'] += buffer['requested_bytes']
        ordered = sorted(groups.items(), key=lambda item: item[1]['allocation_bytes'], reverse=True)
        print(json.dumps({
            'driver_usage_bytes': report['vram_usage_bytes'],
            'driver_usage_measured': report['vram_measured'],
            'renderer_tracked_bytes': report['tracked_device_local_bytes'],
            'untracked_including_simulation_bytes': report['untracked_bytes'],
            'simulation_device_local_bytes': simulation['device_local_bytes'],
            'simulation_allocation_bytes': simulation['allocation_bytes'],
            'simulation_buffers_by_name': [{'name': name, **value} for name, value in ordered],
            'cache': client.call('sim_cache.status'),
        }, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
