#version 450

layout(location = 0) flat in uvec2 vPickId;

// ★★ +1 KAYDIRMA: 0 "hicbir sey" demek, cunku hedef her gecis basinda 0'a
//   temizleniyor. Kimligi oldugu gibi yazmak, 0 numarali mesh'i "bos piksel"
//   ile ayirt edilemez yapardi -- ve bu, bu deponun en pahali hata sinifinin
//   (yokluk ile olculmus deger ayni degere kodlanir) shader tarafindaki hali.
layout(location = 0) out uvec2 outPickId;

void main() {
    outPickId = uvec2(vPickId.x + 1u, vPickId.y);
}
