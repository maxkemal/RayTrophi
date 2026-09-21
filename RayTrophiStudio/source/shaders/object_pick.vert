#version 450

// ★★★★★ SECIM GECISI, ve ana gecisten AYRI olmasi bilincli bir karar:
//   ana render pass'e ikinci bir hedef eklemek oradaki HER pipeline'i
//   (solid, matcap, materyal, transmission, on gecis) degistirirdi. Bu gecis
//   yalnizca TIK aninda kosar ve mevcut hicbir seye dokunmaz.
//
// ★★★★ Konum ifadesi ana gecisle BIREBIR AYNI olmak zorunda: `viewProj * M * p`.
//   Parantezi kaydirmak (`viewProj * (M * p)`) farkli yuvarlama verir ve bu
//   depoda bir kez pahaliya ogrenildi -- derinlik on gecisi ana gecisle
//   invariant degildi ve maskenin derinlik kapisi tutmuyordu. `invariant`
//   burada da beyan ediliyor.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inModelCol0;
layout(location = 2) in vec4 inModelCol1;
layout(location = 3) in vec4 inModelCol2;
layout(location = 4) in vec4 inModelCol3;

layout(push_constant) uniform PickPush {
    mat4 viewProj;
    uint meshId;      // cizim sirasindaki mesh indeksi; CPU tarafi eslemeyi tutar
    uint firstInst;   // bu cizimin global instance tabani
    uint pad0, pad1;
} pc;

// ★★★ Mesh kimligi TEK BASINA yetmez: bir mesh birden cok instance ile
//   cizilebilir (scatter/foliage). Instance yuvasi ayri bir kanalda
//   tasiniyor -- ikisini tek 32 bite paketlemek, instance sayisina sessiz
//   bir tavan koymak olurdu.
layout(location = 0) flat out uvec2 vPickId;

invariant gl_Position;

void main() {
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);
    gl_Position = pc.viewProj * model * vec4(inPosition, 1.0);
    // gl_InstanceIndex firstInstance'i ZATEN icerir; tabani ayrica eklemek
    // instance'i iki kez sayardi.
    vPickId = uvec2(pc.meshId, uint(gl_InstanceIndex));
}
