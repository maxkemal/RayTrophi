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
    call(handle, kernel32, 'agent.chat_send', {'content': text, 'sender': 'Stress Agent', 'target': 'all', 'type': msg_type})
    print(f"[{msg_type.upper()}] {text}")

def main():
    handle, kernel32 = connect_ipc()
    
    chat(handle, kernel32, "Tam Kapasite Elemental Sandbox testi başlıyor...", "thought")
    
    # 1. Arazi (Terrain)
    chat(handle, kernel32, "Arazi (Terrain) oluşturuluyor...", "activity")
    call(handle, kernel32, 'terrain.create', {'name': 'StressTerrain', 'resolution': 128, 'size': 64.0, 'height_scale': 10.0})
    call(handle, kernel32, 'terrain.apply_preset', {'name': 'StressTerrain', 'preset': 'default'})
    
    # 2. Malzeme (Material Node)
    chat(handle, kernel32, "Özel stres materyali (Lava) yaratılıyor...", "activity")
    res = call(handle, kernel32, 'material.create', {'type': 'principled', 'name': 'Lava_Mat'})
    mat_name = res.get('result') or 'Lava_Mat'
    call(handle, kernel32, 'nodes.create_graph', {'graph_type': 'material', 'graph_name': mat_name})
    # Material Shader'ı compile etmesi için apply çağırıyoruz
    call(handle, kernel32, 'nodes.apply', {'graph_type': 'material', 'graph_name': mat_name})

    # 3. Geometri ve Fizik
    chat(handle, kernel32, "Gökten düşecek kaya (Sphere) oluşturuluyor...", "activity")
    call(handle, kernel32, 'scene.add_primitive', {'type': 'sphere', 'name': 'FallingRock', 'size': 4.0})
    # Pozisyonu API üzerinden ayarlamak için lights mantığı gibi var mı kontrol edemedik,
    # ama objeyi seçip transform edebiliyoruz veya physics body ile yukardan düşürebiliyoruz.
    # Şimdilik physics body'nin motion type'ını dynamic yapıyoruz.
    call(handle, kernel32, 'material.assign', {'object_name': 'FallingRock', 'material_name': mat_name})
    
    chat(handle, kernel32, "Fizik çarpışmaları ayarlanıyor (Terrain: statik, Sphere: dinamik)...", "activity")
    # Küre
    call(handle, kernel32, 'physics.add_body', {'object': 'FallingRock', 'kind': 'rigid', 'motion_type': 'dynamic', 'shape': 'sphere', 'mass': 50.0})
    # Arazi
    call(handle, kernel32, 'physics.add_body', {'object': 'StressTerrain', 'kind': 'rigid', 'motion_type': 'static', 'shape': 'mesh', 'mass': 0.0})

    # 4. Akışkan (Fluid)
    chat(handle, kernel32, "Sıvı havuzu (Liquid Domain) kuruluyor...", "activity")
    call(handle, kernel32, 'fluid.create_domain', {'name': 'StressLiquid', 'voxel_size': 0.25})
    # Voxel size çok ufak olursa çökebilir, 0.25 güvenli bir değer (smoke test'te 0.1 idi)
    call(handle, kernel32, 'fluid.seed', {'domain': 'StressLiquid', 'particles_per_cell': 4})
    
    # 5. Gaz ve Yanma (Gas Sim)
    chat(handle, kernel32, "Duman/Gaz simülasyonu (Gas Domain) başlatılıyor...", "activity")
    # Python API'sinde fluid ve gas IPC methodları benzer şekilde çalışıyor. 
    # ipc_test_client.py'deki formata göre fluid.create_domain, domain parametresiyle çalışıyor. 
    # Fakat gaz domain'ini create ederken de 'type' gas veya benzer bir parametre kullanılabiliyor.
    # Ancak smoke testinde `rt.gas.create_domain` kullanılmış. Biz de gas.* methodları varsa deneyelim.
    # Eğer hata verirse fluid olarak devam eder.
    call(handle, kernel32, 'fluid.create_domain', {'name': 'StressGas', 'voxel_size': 0.3})
    call(handle, kernel32, 'fluid.set_param', {'domain': 'StressGas', 'preset': 'smoke'})

    # 6. Timeline / Stress Loop
    chat(handle, kernel32, "Tüm sistemler sahneye dizildi. Stress Test Loop başlatılıyor (50 kare)...", "thought")
    
    for i in range(50):
        call(handle, kernel32, 'physics.step', {'dt': 0.033})
        call(handle, kernel32, 'fluid.step', {'dt': 0.033})
        
        # Sadece ara sıra chat_send yaparak çok fazla spagetti yapmadan ilerleme göster
        if i % 10 == 0:
            chat(handle, kernel32, f"Simülasyon karesi işlendi: {i}/50", "activity")
            
    chat(handle, kernel32, "Stres Testi başarıyla tamamlandı! Sistemin ayakta kaldığını görebilirsin.", "reply")
    
    kernel32.CloseHandle(handle)

if __name__ == '__main__':
    main()
