import json
import ctypes
import ctypes.wintypes as wintypes
import time
import sys
import os

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
        if error != 234:  # ERROR_MORE_DATA
            raise OSError(f"ReadFile failed (error {error})")
    try:
        response_str = b"".join(chunks).decode('utf-8')
        return json.loads(response_str)
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {}

def chat(handle, kernel32, text, msg_type='activity'):
    call(handle, kernel32, 'agent.chat_send', {'content': text, 'sender': 'Wipeout Agent', 'target': 'all', 'type': msg_type})
    print(f"[{msg_type.upper()}] {text}")

def create_dummy_texture(filepath):
    # Çok basit beyaz/gri noise bir png veya sadece solid bir renk oluşturalım
    # Gerekli bağımlılıklara girmemek için Python'ın yerleşik kütüphaneleriyle yapamıyorsak 
    # IPC üzerinden material rengini beyaza çekeriz, ama kullanıcı 'textureler yükle' dediği için:
    try:
        import struct
        import zlib
        width, height = 256, 256
        # Solid white pixel data
        raw_data = b'\x00' + b'\xff\xff\xff\xff' * width
        idat = zlib.compress(raw_data * height)
        def chunk(type_str, data):
            return struct.pack('>I', len(data)) + type_str + data + struct.pack('>I', zlib.crc32(type_str + data) & 0xffffffff)
        
        with open(filepath, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)))
            f.write(chunk(b'IDAT', idat))
            f.write(chunk(b'IEND', b''))
    except Exception as e:
        print(f"Failed to create dummy texture: {e}")

def main():
    handle, kernel32 = connect_ipc()
    
    chat(handle, kernel32, "Kış Kıyamet (Winter Wipeout) senaryosu başlıyor. Kemerleri bağlayın!", "thought")
    
    # 1. Yeni Sahne
    chat(handle, kernel32, "Mevcut sahne siliniyor ve boş bir sahne açılıyor...", "activity")
    call(handle, kernel32, 'templates.open', {'id': 'raytrophi.start.empty', 'conflict_policy': 'discard'})
    time.sleep(1) # Engine'in yeni sahneyi yüklemesi için kısa bir nefes
    
    # 2. Doku & Materyal
    tex_path = os.path.abspath('snow_texture.png')
    create_dummy_texture(tex_path)
    
    chat(handle, kernel32, "Kar dokusu diske yazıldı ve materyale atanıyor...", "activity")
    call(handle, kernel32, 'material.create', {'type': 'principled', 'name': 'SnowMat'})
    # Texture yükleme
    call(handle, kernel32, 'material.set_texture', {'material_name': 'SnowMat', 'slot': 'base_color', 'path': tex_path})
    
    # 3. Arazi
    chat(handle, kernel32, "Deasa arazi (1000m boyut, 1024 çözünürlük) oluşturuluyor...", "thought")
    call(handle, kernel32, 'terrain.create', {'name': 'WinterMountain', 'resolution': 1024, 'size': 1000.0, 'height_scale': 150.0})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'WinterMountain', 'preset': 'mountain'})
    
    chat(handle, kernel32, "Arazi Evaluate ediliyor. Bu işlem asenkron olduğu için bitmesi beklenecek...", "activity")
    call(handle, kernel32, 'terrain.evaluate', {'name': 'WinterMountain'})
    
    # Poll for evaluation
    max_wait = 30
    for i in range(max_wait):
        status = call(handle, kernel32, 'terrain.evaluation_status', {'name': 'WinterMountain'})
        res = status.get('result', {})
        state = res.get('state', 'unknown')
        if state in ('idle', 'completed'):
            chat(handle, kernel32, f"Terrain Evaluate işlemi {i} saniyede tamamlandı!", "activity")
            break
        elif state == 'failed':
            chat(handle, kernel32, "Terrain Evaluate başarısız oldu!", "error")
            break
        time.sleep(1)
        
    # Arazi Materyal ataması API üzerinden doğrudan yoksa, 
    # terrain objesine material assign edelim:
    call(handle, kernel32, 'material.assign', {'object_name': 'WinterMountain', 'material_name': 'SnowMat'})

    # 4. Nehirler
    chat(handle, kernel32, "Araziye nehir kazınıyor...", "activity")
    river_res = call(handle, kernel32, 'terrain.carve_river', {'name': 'WinterMountain', 'river': 'River1'})
    if 'error' in river_res:
        chat(handle, kernel32, f"Nehir kazınırken hata (Beklenen olabilir): {river_res['error']}", "thought")
    else:
        chat(handle, kernel32, "Nehir başarıyla kazındı.", "activity")

    # 5. Biome & Ağaçlar
    chat(handle, kernel32, "Ağaç (Tree) objesi oluşturuluyor...", "activity")
    call(handle, kernel32, 'scene.add_primitive', {'type': 'cylinder', 'name': 'TreeAsset', 'size': 5.0})
    
    chat(handle, kernel32, "ForestBiome grubu yaratılıp ağaçlar araziye yayılıyor (Scatter Fill)...", "thought")
    call(handle, kernel32, 'scatter.create_group', {'name': 'ForestBiome', 'target_node': 'WinterMountain', 'target_type': 'terrain'})
    call(handle, kernel32, 'scatter.add_source', {'group': 'ForestBiome', 'mesh': 'TreeAsset', 'weight': 1.0})
    
    # Yoğunluğu artırmak için settings (varsayılanı da kullanabiliriz)
    call(handle, kernel32, 'scatter.set_settings', {'group': 'ForestBiome', 'target_count': 10000})
    
    fill_res = call(handle, kernel32, 'scatter.fill', {'group': 'ForestBiome'})
    instance_count = fill_res.get('result', 0)
    
    if 'error' in fill_res:
        chat(handle, kernel32, f"Scatter Fill hatası: {fill_res['error']}", "error")
    else:
        chat(handle, kernel32, f"İnanılmaz! Araziye tam {instance_count} adet ağaç dikildi.", "reply")

    chat(handle, kernel32, "Kış Kıyamet testi sağ salim bitti. Sahnenin tadını çıkar!", "reply")

    kernel32.CloseHandle(handle)

if __name__ == '__main__':
    main()
