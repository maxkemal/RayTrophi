struct ScreenGiPixel { vec4 light; vec4 surface; vec4 aux; };
layout(set=0,binding=24,std430) readonly buffer ScreenGi {
    uvec4 screenGiMeta; // magic, width, height, samples
    ScreenGiPixel screenGiPixels[];
};
// Yuzey kimligi kapisi: bu GI pikseli GERCEKTEN bu fragment'e mi ait.
// Iki tuketici de ayni kapiyi kullanmak ZORUNDA, yoksa biri otekinin
// reddettigi bir olcumu kabul eder ve fark hicbir yerde gorunmez.
bool screenGiPixelMatches(vec3 normal, out ScreenGiPixel s) {
    s.light=vec4(0); s.surface=vec4(0,0,0,1); s.aux=vec4(0);
    if (screenGiMeta.x!=0x53474932u || (vMaterialID&0x80000000u)!=0u) return false;
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(screenGiMeta.yz)))) return false;
    s=screenGiPixels[uint(pixel.y)*screenGiMeta.y+uint(pixel.x)];
    if (s.surface.w>=1.0 || abs(s.surface.w-gl_FragCoord.z)>2e-7) return false;
    // The prototype reconstructs geometric normals; sharply different mapped
    // or smooth normals use the established fallback instead of wrong lighting.
    return dot(normal,s.surface.xyz)>=0.5;
}

// ★★★★★ OLCULEN gokyuzu gorunurlugu. `sampleScreenGi`den AYRI bir fonksiyon
//   ve ayri bir guven tasir: bir piksel hicbir ornegini golgelendiremeyip
//   (light.w==0) yine de yarim kuresinin ne kadarinin duvarla kapali oldugunu
//   TAM olarak biliyor olabilir. Bu iki durumu tek cagriya katlamak, engellenen
//   isinlari "olcum yok"a cevirir -- ve tuketici orayi ENGELSIZ gokyuzuyle
//   doldurdugu icin oda mavi ile dolar.
// false donmesi "gorus acik" DEMEK DEGILDIR, "olcemedim" demektir.
bool sampleScreenGiSkyVisibility(vec3 normal, out float visibility) {
    visibility=1.0;
    if (screenGiMeta.x!=0x53474932u || (vMaterialID&0x80000000u)!=0u) return false;
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(screenGiMeta.yz)))) return false;
    ScreenGiPixel s=screenGiPixels[uint(pixel.y)*screenGiMeta.y+uint(pixel.x)];
    if (s.surface.w>=1.0 || s.aux.y<=0.0) return false;

    // ★★★★★ DERINLIK TOLERANSI BURADA GORUNTU-BAGIMLI, ve radyans okumasindan
    //   BILEREK daha gevsek. Sabit 2e-7'lik kapi (radyans kolunda hala oyle)
    //   EGIK yuzeylerde tutmuyor: bir duvara veya tavana dar aciyla bakildiginda
    //   derinlik piksel basina hizla degisir, interpole edilen `gl_FragCoord.z`
    //   ile GI tamponundaki dokudan okunmus derinlik 2e-7'yi kolayca asar.
    //   Kameraya bakan mobilyada asmaz -- ve olculen belirti tam olarak buydu:
    //   mobilya duzeldi, duvar ve tavan mavi kaldi.
    //
    // ★★★ Iki okuyucunun FARKLI siki olmasi gerekiyor cunku TASIDIKLARI RISK
    //   farkli. Komsu pikselin RADYANSINI bu yuzeye tasimak gorunur bir isik
    //   sizintisidir; komsu pikselin GOKYUZU GORUNURLUGUNU tasimak, yumusak ve
    //   yavas degisen bir alanda kucuk bir hatadir. Ayni esigi paylasmalari
    //   "tutarli" degil, yalnizca birinin digerinin riskini odemesiydi.
    //
    // ★★ Tolerans `fwidth` ile OLCULUR, uydurulmaz: bir pikselluk gercek
    //   derinlik degisimi. Duz yuzeyde kucuk (siki kalir), egik yuzeyde buyuk
    //   (kabul eder), siluet kenarinda buyuk -- orada biraz sizar, ve bu
    //   gorunurluk gibi yumusak bir alan icin kabul edilebilir bir takas.
    const float depthSlack=max(2e-7,fwidth(gl_FragCoord.z)*2.0);
    if (abs(s.surface.w-gl_FragCoord.z)>depthSlack) return false;
    if (dot(normal,s.surface.xyz)<0.5) return false;

    // ★★★ `aux.z == 1` bu degerin KOMSUDAN DOLDURULDUGUNU soyler (filtre
    //   gecisi, kendi yarim kure verdikti olmayan piksel icin). Burada
    //   KABUL EDILIYOR ve bu bilincli bir takas: alternatifi 1.0'a dusmek,
    //   yani "olcemedim"i "gokyuzune tamamen acik"a cevirmekti -- mumkun
    //   olan en parlak cevap. Komsunun gorunurlugu yumusak bir alanda
    //   kucuk bir hata; engelsiz gokyuzu ise gorunur bir ariza.
    // ★★ Radyans okuyucusu (`screenGiPixelMatches`) bu degeri KULLANMAZ:
    //   komsudan radyans tasimak gorunur bir isik sizintisi olurdu.
    visibility=clamp(s.aux.x,0.0,1.0);
    return true;
}

bool sampleScreenGi(vec3 normal, out vec3 irradiance, out float confidence) {
    irradiance=vec3(0);
    confidence=0.0;
    ScreenGiPixel s;
    if (!screenGiPixelMatches(normal,s)) return false;
    if (s.light.w<=0.0) return false;
    irradiance=s.light.rgb;
    confidence=clamp(s.light.w,0.0,1.0);
    return true;
}
