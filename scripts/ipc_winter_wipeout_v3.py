import json
import ctypes
import ctypes.wintypes as wintypes
import time
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'

def connect_ipc():
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateFileW(PIPE_NAME, 0xC0000000, 0, None, 3, 0, None)
    if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
        print('Failed to connect to IPC. Is RayTrophi Studio running?')
        sys.exit(1)
    
    mode = wintypes.DWORD(2)
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None)
    return handle, kernel32

def call(handle, kernel32, method, params={}):
    msg = json.dumps({'id': 1, 'method': method, 'params': params}).encode('utf-8')
    written = wintypes.DWORD(0)
    kernel32.WriteFile(handle, msg, len(msg), ctypes.byref(written), None)
    
    chunks = []
    while True:
        buf = ctypes.create_string_buffer(65536)
        read = wintypes.DWORD(0)
        success = kernel32.ReadFile(handle, buf, 65536, ctypes.byref(read), None)
        chunks.append(buf.raw[:read.value])
        if success:
            break
        error = kernel32.GetLastError()
        if error != 234:
            raise OSError(f"ReadFile failed (error {error})")
    try:
        response_str = b"".join(chunks).decode('utf-8')
        return json.loads(response_str)
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {}

def chat(handle, kernel32, text, msg_type='activity'):
    call(handle, kernel32, 'agent.chat_send', {'content': text, 'sender': 'Smart Agent v3', 'target': 'all', 'type': msg_type})
    print(f"[{msg_type.upper()}] {text}")

def main():
    handle, kernel32 = connect_ipc()
    
    chat(handle, kernel32, "Kış Kıyamet v3 başlıyor! C++ API güncellendi, Biome ve Foliage yetenekleri devrede.", "thought")
    
    chat(handle, kernel32, "Sahne sıfırlanıyor...", "activity")
    call(handle, kernel32, 'templates.open', {'id': 'raytrophi.start.empty', 'conflict_policy': 'discard'})
    time.sleep(1)
    
    chat(handle, kernel32, "1024 çözünürlük, 1000m dev arazi oluşturuluyor...", "activity")
    call(handle, kernel32, 'terrain.create', {'name': 'UltimateMountain', 'resolution': 1024, 'size': 1000.0, 'height_scale': 150.0})
    
    # Adım Adım Preset Uygulamaları (False = replace_graph kapalı, çünkü üst üste ekleyeceğiz)
    chat(handle, kernel32, "1. Dağ topolojisi ve erozyon uygulanıyor (snowy_mountain_valley)...", "activity")
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'snowy_mountain_valley', 'replace_graph': True})
    
    chat(handle, kernel32, "2. Nehir ağı (river_network) grafiğe ekleniyor...", "activity")
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'river_network', 'replace_graph': False})
    
    chat(handle, kernel32, "3. Biyom maskeleri (biome_boreal) grafiğe ekleniyor...", "activity")
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'biome_boreal', 'replace_graph': False})
    
    chat(handle, kernel32, "4. Varlıklar Kütüphanesinden Ağaçlar (biome_foliage) çekiliyor...", "activity")
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'biome_foliage', 'replace_graph': False})
    
    chat(handle, kernel32, "Tüm presetler grafiğe bağlandı. Şimdi devasa asenkron hesaplama (Evaluate) başlıyor...", "thought")
    call(handle, kernel32, 'terrain.evaluate', {'name': 'UltimateMountain'})
    
    for i in range(30):
        status = call(handle, kernel32, 'terrain.evaluation_status', {'name': 'UltimateMountain'})
        state = status.get('result', {}).get('state', 'unknown')
        if state in ('idle', 'completed'):
            chat(handle, kernel32, f"Hesaplama {i} saniyede tamamlandı! Dağ, Nehirler ve Boreal Ormanı şu an ekranda olmalı.", "reply")
            break
        time.sleep(1)
            
    kernel32.CloseHandle(handle)

if __name__ == '__main__':
    main()
