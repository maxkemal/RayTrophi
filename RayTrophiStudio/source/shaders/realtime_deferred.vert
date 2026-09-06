// ===========================================================================
// KULLANILMIYOR - REFERANS (2026-08-31)
// Ayri "Realtime" deferred viewport modu sokuldu; bu shader'i yukleyen kod
// artik yok. Gerekce: docs/dev/REALTIME_RENDERER_ROADMAP.md
// Deferred'a donuldugunde baslangic noktasi olsun diye birakildi.
// UYARI: eski hali depth image'i input attachment olarak okuyordu, ama
// depthImage INPUT_ATTACHMENT usage biti olmadan olusturuluyor -- geri
// baglayan kisi once orayi duzeltmeli.
// ===========================================================================
#version 450

layout(location = 0) out vec2 vTexCoord;

void main() {
    // Fullscreen triangle
    vTexCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vTexCoord * 2.0 - 1.0, 0.0, 1.0);
}
