import json
import ctypes
import ctypes.wintypes as wintypes
import time
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
kernel32 = ctypes.windll.kernel32
handle = kernel32.CreateFileW(PIPE_NAME, 0xC0000000, 0, None, 3, 0, None)
if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
    print('Failed to connect')
    sys.exit(1)

mode = wintypes.DWORD(2)
kernel32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None)

def call(method, params={}):
    msg = json.dumps({'id': 1, 'method': method, 'params': params}).encode('utf-8')
    written = wintypes.DWORD(0)
    kernel32.WriteFile(handle, msg, len(msg), ctypes.byref(written), None)
    buf = ctypes.create_string_buffer(65536)
    read = wintypes.DWORD(0)
    kernel32.ReadFile(handle, buf, 65536, ctypes.byref(read), None)
    try:
        return json.loads(buf.raw[:read.value].decode('utf-8'))
    except:
        return {}

def chat(text, msg_type='thought'):
    call('agent.chat_send', {'content': text, 'sender': 'Apprentice Agent', 'target': 'all', 'type': msg_type})
    time.sleep(1.5)

chat('Uyandım ve yeni bir dünyaya (RayTrophi Studio) bağlandım. Neler yapabildiğimi öğrenmek için sistemi keşfetmeye başlıyorum...', 'thought')

# 1. Discover
res = call('agent.discover')
discover_data = res.get('result', {})
chat(f'agent.discover komutunu çalıştırdım. Kayıtlı IPC metod sayısı: {discover_data.get("registered_methods")}. Dokümantasyon kapsama oranı: %{int(discover_data.get("documented_coverage", 0)*100)}.', 'activity')

# 2. State Summary
res = call('agent.get_state_summary', {'include_probe': True})
state_data = res.get('result', {})
chat(f'Şu anki sahne durumunu okudum. Aktif proje: {state_data.get("project", {}).get("path", "Bilinmiyor")}. Seçili obje sayısı: {len(state_data.get("selection", []))}.', 'thought')

# 3. List Methods
res = call('agent.list_methods')
methods = res.get('result', [])
chat(f'agent.list_methods çalıştırdım. Toplamda {len(methods)} farklı IPC komutu buldum. Sahne oluşturmak için scene.add_primitive komutunu inceleyeceğim.', 'activity')

# 4. Describe a method
res = call('agent.describe', {'method': 'scene.add_primitive'})
desc = res.get('result', {})
params_info = ", ".join([p['name'] + ' (' + p['type'] + ')' for p in desc.get('parameters', [])])
chat(f'scene.add_primitive için parametreleri öğrendim: {params_info}.', 'thought')

# 5. Take action! Let's add a primitive.
chat('Şimdi öğrendiğim komutla sahneye bir torus (halka) eklemeyi deneyeceğim...', 'thought')
res = call('scene.add_primitive', {'type': 'torus', 'name': 'AgentTorus', 'size': 2.0})
if 'error' not in res:
    chat('Başardım! Sahneye "AgentTorus" adında bir obje ekledim. UI üzerinden kontrol edebilirsin! Yeni araçlar öğrenmeye hazırım.', 'reply')
else:
    chat(f'Hata aldım: {res.get("error")}', 'error')
