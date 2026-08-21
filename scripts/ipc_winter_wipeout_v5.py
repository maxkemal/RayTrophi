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
    call(handle, kernel32, 'agent.chat_send', {'content': text, 'sender': 'Smart Agent v5', 'target': 'all', 'type': msg_type})
    print(f"[{msg_type.upper()}] {text}")

def main():
    handle, kernel32 = connect_ipc()
    
    chat(handle, kernel32, "Kış Kıyamet v5: Otonom Asset Kütüphanesi Araması başlıyor!", "thought")
    
    chat(handle, kernel32, "Sahne sıfırlanıyor...", "activity")
    call(handle, kernel32, 'templates.open', {'id': 'raytrophi.start.empty', 'conflict_policy': 'discard'})
    time.sleep(1)
    
    chat(handle, kernel32, "Asset Kütüphanesindeki varlıklar IPC üzerinden sorgulanıyor...", "thought")
    assets_resp = call(handle, kernel32, 'scatter.list_assets')
    assets = assets_resp.get('result', [])
    
    chat(handle, kernel32, f"Kütüphanede tam {len(assets)} adet foliage varlığı bulundu!", "activity")
    
    # Kendi zekamızla (keyword filtreleme) en uygun ağacı bulalım:
    keywords = ["çam", "agac", "pine", "fir", "spruce", "tree"]
    selected_asset = None
    
    for a in assets:
        name_lower = a.get('name', '').lower()
        # Sadece vegetation (bitki/ağaç) kategorisindekileri al
        if a.get('category', '').lower() in ['vegetation', 'trees', 'nature', 'bitki', 'agac', '']:
            for kw in keywords:
                if kw in name_lower:
                    selected_asset = a
                    break
        if selected_asset:
            break
            
    # Eğer filtreye uyan bir şey bulamazsak listedeki ilk asset'i alalım
    if not selected_asset and len(assets) > 0:
        selected_asset = assets[0]
        
    if not selected_asset:
        chat(handle, kernel32, "HATA: Asset Kütüphanesi tamamen boş! Lütfen en az bir ağaç modeli ekleyin.", "error")
        return
        
    chat(handle, kernel32, f"Otonom zeka karar verdi: Ormana '{selected_asset['name']}' adlı gerçek Asset ekilecek!", "reply")
    
    chat(handle, kernel32, "1024 çözünürlük, 1000m dev arazi (UltimateMountain) oluşturuluyor...", "activity")
    call(handle, kernel32, 'terrain.create', {'name': 'UltimateMountain', 'resolution': 1024, 'size': 1000.0, 'height_scale': 150.0})
    
    chat(handle, kernel32, "'Orman' adında Scatter Grubu yaratılıyor ve seçilen Asset bağlanıyor...", "thought")
    call(handle, kernel32, 'scatter.create_group', {'name': 'Orman', 'target_node': 'UltimateMountain', 'target_type': 'terrain'})
    
    # Sahte primitif silindir yerine gerçek Asset Kütüphanesinden yol gösteriyoruz!
    call(handle, kernel32, 'scatter.add_library_source', {'group': 'Orman', 'relative_path': selected_asset['relative_path']})
    call(handle, kernel32, 'scatter.set_settings', {'group': 'Orman', 'target_count': 5000})

    chat(handle, kernel32, "Procedural Node Presetleri üst üste uygulanıyor...", "activity")
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'snowy_mountain_valley', 'replace_graph': True})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'river_network', 'replace_graph': False})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'biome_boreal', 'replace_graph': False})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'UltimateMountain', 'preset': 'biome_foliage', 'replace_graph': False})
    
    chat(handle, kernel32, "Evaluate işlemi (Maske hesaplama) başladı...", "thought")
    call(handle, kernel32, 'terrain.evaluate', {'name': 'UltimateMountain'})
    
    for i in range(30):
        status = call(handle, kernel32, 'terrain.evaluation_status', {'name': 'UltimateMountain'})
        state = status.get('result', {}).get('state', 'unknown')
        if state in ('idle', 'completed'):
            break
        time.sleep(1)
        
    chat(handle, kernel32, "Hesaplama bitti. Gerçek Asset (Ağaç) Scatter Fill (Serpme) yapılıyor!", "activity")
    fill_res = call(handle, kernel32, 'scatter.fill', {'group': 'Orman'})
    
    if 'error' in fill_res:
        chat(handle, kernel32, f"Scatter Fill Hata: {fill_res['error']}", "error")
    else:
        # Check if the result was a dict containing 'spawned'
        result = fill_res.get('result')
        if isinstance(result, dict):
            instance_count = result.get('spawned', 0)
        else:
            instance_count = result if result else 0
        chat(handle, kernel32, f"V5 BAŞARILI! Dağ yüzeyindeki Boreal orman maskesine göre {instance_count} adet gerçek Asset ('{selected_asset['name']}') mükemmel şekilde dikildi.", "reply")
            
    kernel32.CloseHandle(handle)

if __name__ == '__main__':
    main()
