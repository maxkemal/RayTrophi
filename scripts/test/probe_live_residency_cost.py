"""Measure the user's open scene via external IPC; no reset, build, or launch."""
import argparse,json,time,sys,ctypes
from pathlib import Path
from ctypes import wintypes
from rt_ipc import RtIpc


def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    # Resolve pipe server PID rather than relying on an executable name.
    pid=wintypes.DWORD()
    if not ctypes.windll.kernel32.GetNamedPipeServerProcessId(client.handle,ctypes.byref(pid)):
        return {}
    k=ctypes.windll.kernel32;k.OpenProcess.restype=wintypes.HANDLE
    h=k.OpenProcess(0x410,False,pid.value)
    if not h:return {'pid':pid.value}
    c=Counters();c.cb=ctypes.sizeof(c)
    try:
        if ctypes.windll.psapi.GetProcessMemoryInfo(h,ctypes.byref(c),c.cb):
            return {'pid':pid.value,'working_set_bytes':c.WorkingSetSize,'private_bytes':c.PrivateUsage}
        return {'pid':pid.value}
    finally:k.CloseHandle(h)

parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=8);parser.add_argument('--gpu-timing',action='store_true')
args=parser.parse_args();client=RtIpc();root=Path(__file__).resolve().parents[2]
out=root/'docs/dev/gas_residency_timeline_probe_2026_10_10.json'
data=json.loads(out.read_text()) if out.exists() else {'rows':[]}
control=client.call('sim.control_state');assert not control['playing'],'Pause before measurement'
initial=client.call('perf.gpu_kernel_timings');enabled=initial['enabled']
try:
    if args.gpu_timing:client.call('perf.set_gpu_kernel_timing',enabled=True);client.call('perf.gpu_kernel_timings',reset=True)
    for _ in range(args.frames):
        class AvailableMemory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
        available=AvailableMemory();available.length=ctypes.sizeof(available)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(available))
        if available.available < 12 * 1024 ** 3 or available.page_available < 8 * 1024 ** 3:
            print('STOP: memory headroom guard',flush=True)
            break
        frame=client.call('sim.control_state')['frame']+1
        t=time.perf_counter();result=client.call('timeline.set_frame',frame=frame);wall=(time.perf_counter()-t)*1000
        # Separate calls allow the application's frame loop to finish the scrub.
        actual=client.call('sim.control_state')
        row={'requested_frame':frame,'control':actual,'wall_ms':wall,'process':memory(),
             'cache':client.call('sim_cache.status'),'fluid':client.call('fluid.step_stats',domain='Physics Domain 1'),
             'gas':client.call('gas.step_stats',domain='Physics Domain 1')}
        row['observed_step_and_queries_ms']=(time.perf_counter()-t)*1000
        if args.gpu_timing:row['kernels']=client.call('perf.gpu_kernel_timings',reset=True)
        data['rows'].append(row);data['gpu_memory']=client.call('perf.get_gpu_memory')
        out.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps({'frame':actual['frame'],'wall_ms':round(wall,2),'fluid_ms':row['fluid'].get('total_ms'),'gas_ms':row['gas'].get('total_ms'),'surface_ms':row['gas'].get('gpu_surface_exchange_ms'),'inventory_ms':row['gas'].get('inventory_advection_ms'),'snapshot':row['gas'].get('inventory_snapshot_download_bytes'),'cache_bytes':row['cache']['ram_bytes'],'ram':row['process']}),flush=True)
finally:
    if args.gpu_timing and not enabled:client.call('perf.set_gpu_kernel_timing',enabled=False)
    client.close()
