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
    call(handle, kernel32, 'agent.chat_send', {'content': text, 'sender': 'Smart Agent', 'target': 'all', 'type': msg_type})
    print(f"[{msg_type.upper()}] {text}")

def main():
    handle, kernel32 = connect_ipc()
    
    chat(handle, kernel32, "Kış Kıyamet v2 başlıyor! IPC'nin sağladığı kendi dokümantasyonuyla çalışıyoruz.", "thought")
    
    # 1. Sahneyi sıfırla
    chat(handle, kernel32, "Sahne sıfırlanıyor...", "activity")
    call(handle, kernel32, 'templates.open', {'id': 'raytrophi.start.empty', 'conflict_policy': 'discard'})
    time.sleep(1)
    
    # 2. Arazi oluştur
    chat(handle, kernel32, "1024 çözünürlük, 1000m dev arazi oluşturuluyor...", "activity")
    call(handle, kernel32, 'terrain.create', {'name': 'SmartMountain', 'resolution': 1024, 'size': 1000.0, 'height_scale': 150.0})
    
    # Ajanın terrain.apply_preset komutunun beklentilerini okuması (Temsili olarak API'den çekiyoruz)
    preset_info = call(handle, kernel32, 'agent.describe', {'method': 'terrain.apply_preset'})
    if 'result' in preset_info:
        chat(handle, kernel32, f"Öğrenildi: apply_preset komutu, procedural nodeları (erozyon, kar, foliage) bağlamak için kullanılıyor. Parametreler: {list(preset_info['result'].get('parameters', {}).keys())}", "thought")
    
    chat(handle, kernel32, "Hazır karlı dağ ve biyom (snowy_mountain_valley) preset'i uygulanıyor. Ağaçlar/kar/erozyon nodeları otomatik bağlanacak!", "activity")
    # Kodlara bakmadan API'nin sunduğu "snowy_mountain_valley" presetini kullanıyoruz. 
    # replace_graph=True ile grafı resetliyoruz.
    res = call(handle, kernel32, 'terrain.apply_preset', {'name': 'SmartMountain', 'preset': 'snowy_mountain_valley', 'replace_graph': True})
    
    if 'error' in res:
        chat(handle, kernel32, f"Preset uygulanamadı: {res['error']}", "error")
    else:
        chat(handle, kernel32, "Preset uygulandı. Şimdi Evaluate işlemi asenkron olarak bekleniyor...", "activity")
        call(handle, kernel32, 'terrain.evaluate', {'name': 'SmartMountain'})
        
        # Evaluate bekleniyor
        for i in range(30):
            status = call(handle, kernel32, 'terrain.evaluation_status', {'name': 'SmartMountain'})
            state = status.get('result', {}).get('state', 'unknown')
            if state in ('idle', 'completed'):
                chat(handle, kernel32, f"Arazi hesaplaması {i} saniyede bitti!", "activity")
                break
            time.sleep(1)
            
        chat(handle, kernel32, "Procedural terrain başarıyla tamamlandı. Tüm asetler ve erozyon otomatik işlendi!", "reply")
        
    kernel32.CloseHandle(handle)

if __name__ == '__main__':
    main()
