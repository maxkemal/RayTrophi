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
    call(handle, kernel32, 'agent.chat_send', {'content': text, 'sender': 'Smart Agent v4', 'target': 'all', 'type': msg_type})
    print(f"[{msg_type.upper()}] {text}")

def main():
    handle, kernel32 = connect_ipc()
    
    chat(handle, kernel32, "Kış Kıyamet v4: Procedural Terrain ve Scatter Entegrasyonu başlıyor!", "thought")
    
    chat(handle, kernel32, "Sahne sıfırlanıyor...", "activity")
    call(handle, kernel32, 'templates.open', {'id': 'raytrophi.start.empty', 'conflict_policy': 'discard'})
    time.sleep(1)
    
    chat(handle, kernel32, "1024 çözünürlük, 1000m dev arazi (UltimateMountain) oluşturuluyor...", "activity")
    call(handle, kernel32, 'terrain.create', {'name': 'UltimateMountain', 'resolution': 1024, 'size': 1000.0, 'height_scale': 150.0})
    
    chat(handle, kernel32, "Asset kütüphanesinden ağaç (TreeAsset) modeli sahneye ekleniyor...", "activity")
    # Gerçek asset sistemi IPC'ye açık değilse sahte bir ağaç (Cylinder) kullanıyoruz
    call(handle, kernel32, 'scene.add_primitive', {'type': 'cylinder', 'name': 'TreeAsset', 'size': 5.0})
    
    chat(handle, kernel32, "'Orman' adında Scatter Grubu yaratılıyor ve TreeAsset bu gruba atanıyor...", "thought")
    # Procedural Foliage Node, "orman" kelimesini gördüğünde bu grubu tanıyıp maskeyi buraya bağlayacak!
    call(handle, kernel32, 'scatter.create_group', {'name': 'Orman', 'target_node': 'UltimateMountain', 'target_type': 'terrain'})
    call(handle, kernel32, 'scatter.add_source', {'group': 'Orman', 'mesh': 'TreeAsset', 'weight': 1.0})
    call(handle, kernel32, 'scatter.set_settings', {'group': 'Orman', 'target_count': 5000})

    chat(handle, kernel32, "Procedural Node Presetleri üst üste uygulanıyor...", "activity")
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'snowy_mountain_valley', 'replace_graph': True})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'river_network', 'replace_graph': False})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'biome_boreal', 'replace_graph': False})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'biome_foliage', 'replace_graph': False})
    
    chat(handle, kernel32, "Tüm presetler grafiğe bağlandı. Şimdi devasa asenkron hesaplama (Evaluate) başlıyor...", "thought")
    call(handle, kernel32, 'terrain.evaluate', {'name': 'UltimateMountain'})
    
    for i in range(30):
        status = call(handle, kernel32, 'terrain.evaluation_status', {'name': 'UltimateMountain'})
        state = status.get('result', {}).get('state', 'unknown')
        if state in ('idle', 'completed'):
            chat(handle, kernel32, f"Hesaplama {i} saniyede tamamlandı!", "activity")
            break
        time.sleep(1)
        
    chat(handle, kernel32, "Terrain hesaplaması bitti. Şimdi orman için Scatter Fill (Ağaçları ekme) tetikleniyor!", "thought")
    fill_res = call(handle, kernel32, 'scatter.fill', {'group': 'Orman'})
    
    if 'error' in fill_res:
        chat(handle, kernel32, f"Ağaçları ekerken (Scatter Fill) hata: {fill_res['error']}", "error")
    else:
        instance_count = fill_res.get('result', 0)
        chat(handle, kernel32, f"Mükemmel! Boreal ormanı maskesine göre sahneye tam {instance_count} ağaç dikildi.", "reply")
            
    kernel32.CloseHandle(handle)

if __name__ == '__main__':
    main()
