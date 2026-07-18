/* RayTrophi Studio Manual - Main Script & Interactive Demos */

// Global State
const state = {
    lang: 'en',
    currentHash: '#home',
    searchQuery: '',
    viewportMode: 'home'
};

// Translations Dictionary
const translations = {
    en: {
        title: "RayTrophi Studio",
        status: "Active Development",
        tag: "OptiX & Vulkan RT DCC Engine",
        search: "Search Manual...",
        chapters: "Documentation Chapters",
        quickStart: "Quick Start",
        copied: "Copied to clipboard!",
        clip: "CLIPPING WARNING",
        clipDesc: "Some pixels are overexposed in HDR space"
    },
    tr: {
        title: "RayTrophi Studio",
        status: "Aktif Geliştirme",
        tag: "OptiX & Vulkan RT DCC Motoru",
        search: "Kılavuzda Ara...",
        chapters: "Dökümantasyon Bölümleri",
        quickStart: "Hızlı Başlangıç",
        copied: "Panoya kopyalandı!",
        clip: "PİKSEL PATLAMASI UYARISI",
        clipDesc: "HDR uzayında bazı pikseller aşırı pozlandı"
    }
};

// Mode configuration for titles
const viewportConfig = {
    home: {
        titleEn: "Live Viewport - Default Scene (3 Spheres Path-Tracer)",
        titleTr: "Canlı Viewport - Varsayılan Sahne (3 Küre Işın İzleyici)",
        modeEn: "Default (Overview)",
        modeTr: "Varsayılan (Genel Bakış)"
    },
    world: {
        titleEn: "Live Viewport - Atmospheric Fog & Dynamic Sun elevation",
        titleTr: "Canlı Viewport - Atmosferik Sis & Dinamik Güneş Açısı",
        modeEn: "Atmosphere & Fog",
        modeTr: "Atmosfer & Sis"
    },
    water: {
        titleEn: "Live Viewport - Ocean Wave grid & Caustic reflections",
        titleTr: "Canlı Viewport - Okyanus Dalga Ağı & Kostik Yansımaları",
        modeEn: "FFT Ocean Waves",
        modeTr: "FFT Okyanus Dalgaları"
    },
    nodes: {
        titleEn: "Live Viewport - Procedural Noise heightfield grid",
        titleTr: "Canlı Viewport - Prosedürel Gürültü Yükseklik Haritası",
        modeEn: "Terrain Nodes Output",
        modeTr: "Arazi Düğüm Çıktısı"
    },
    terrain: {
        titleEn: "Live Viewport - Erosion & Scattered tree instances",
        titleTr: "Canlı Viewport - Erozyon & Dağıtılmış Ağaç Örnekleri",
        modeEn: "Instanced Foliage",
        modeTr: "Örneklendirilmiş Bitki Örtüsü"
    },
    "edit-mesh": {
        titleEn: "Live Viewport - Non-destructive wireframe topology edit layers",
        titleTr: "Canlı Viewport - Tahribatsız Kafes Topoloji Düzenleme Katmanı",
        modeEn: "Topology Wireframe",
        modeTr: "Topoloji Tel Kafes"
    },
    sculpt: {
        titleEn: "Live Viewport - Click & drag directly on the sphere to SCULPT!",
        titleTr: "Canlı Viewport - Şekillendirmek için doğrudan küreye TIKLAYIP SÜRÜKLEYİN!",
        modeEn: "Interactive Sculpting",
        modeTr: "İnteraktif Heykel"
    },
    "mesh-paint": {
        titleEn: "Live Viewport - Click & drag directly on the sphere to PAINT colors!",
        titleTr: "Canlı Viewport - Boyamak için doğrudan küreye TIKLAYIP SÜRÜKLEYİN!",
        modeEn: "Interactive Texture Painting",
        modeTr: "İnteraktif Doku Boyama"
    },
    hair: {
        titleEn: "Live Viewport - LSS Hair Groom curves & Marschner shader",
        titleTr: "Canlı Viewport - LSS Saç Eğrileri & Marschner Gölgelendirici",
        modeEn: "Groom Curves",
        modeTr: "Saç Groom Eğrileri"
    },
    animation: {
        titleEn: "Live Viewport - Skeletal rig bones bone joint rotation playback",
        titleTr: "Canlı Viewport - İskelet Kemik Animasyonu Eklem Oynatımı",
        modeEn: "Skeletal Skeleton Rig",
        modeTr: "İskelet Sistemi Yapısı"
    },
    render: {
        titleEn: "Live Viewport - Backend comparison (Left: Vulkan RT, Right: OptiX OIDN)",
        titleTr: "Canlı Viewport - Motor Karşılaştırma (Sol: Vulkan RT, Sağ: OptiX OIDN)",
        modeEn: "Vulkan RT vs NVIDIA OptiX",
        modeTr: "Vulkan RT ve NVIDIA OptiX"
    },
    camera: {
        titleEn: "Live Viewport - Pro Camera HUD (Histogram/Peaking/Zebra)",
        titleTr: "Canlı Viewport - Kamera HUD (Histogram/Peaking/Zebra)",
        modeEn: "Viewfinder Diagnostics",
        modeTr: "Vizör Analiz Göstergeleri"
    },
    system: {
        titleEn: "Live Viewport - RTP Serialization & System Console logging",
        titleTr: "Canlı Viewport - RTP Serileştirme & Sistem Konsol Günlüğü",
        modeEn: "RTP Console Logger",
        modeTr: "RTP Konsol Günlükçüsü"
    }
};

// Paint & Sculpt Texture Arrays
const paintTextureWidth = 128;
const paintTextureHeight = 128;
const paintTexture = new Uint8Array(paintTextureWidth * paintTextureHeight * 3); // RGB
paintTexture.fill(160); // Default grey base

const sculptDisplacement = new Float32Array(paintTextureWidth * paintTextureHeight); // Heights displacement
sculptDisplacement.fill(0.0);

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    initRouter();
    initLanguage();
    initSearch();
    initCodeBlocks();
    initDemos();
});

// ═══════════════════════════════════════════════════════════
// ROUTER & NAVIGATION
// ═══════════════════════════════════════════════════════════
function initRouter() {
    const handleRoute = () => {
        let hash = window.location.hash || '#home';
        state.currentHash = hash;
        
        // Extract mode name
        const mode = hash.substring(1) || 'home';
        state.viewportMode = mode;
        
        // Hide all sections, show active
        document.querySelectorAll('.doc-section').forEach(section => {
            section.classList.remove('active');
        });
        
        const activeSection = document.querySelector(hash);
        if (activeSection) {
            activeSection.classList.add('active');
        } else {
            document.querySelector('#home').classList.add('active');
        }
        
        // Update navigation link states
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === hash) {
                link.classList.add('active');
            }
        });

        // Update persistent viewport title and mode HUD
        const conf = viewportConfig[mode] || viewportConfig['home'];
        
        const titleEn = document.getElementById('viewport-title-en');
        const titleTr = document.getElementById('viewport-title-tr');
        const modeEn = document.getElementById('viewport-mode-en');
        const modeTr = document.getElementById('viewport-mode-tr');

        if (titleEn) titleEn.textContent = conf.titleEn;
        if (titleTr) titleTr.textContent = conf.titleTr;
        if (modeEn) modeEn.textContent = conf.modeEn;
        if (modeTr) modeTr.textContent = conf.modeTr;
        
        // Scroll content to top
        const scroller = document.getElementById('content-scroller');
        if (scroller) scroller.scrollTop = 0;
    };

    window.addEventListener('hashchange', handleRoute);
    handleRoute(); // Run initially
}

// ═══════════════════════════════════════════════════════════
// TRANSLATION / LANGUAGE TOGGLE
// ═══════════════════════════════════════════════════════════
function initLanguage() {
    const toggleBtns = document.querySelectorAll('.lang-btn');
    
    const applyLang = (lang) => {
        state.lang = lang;
        document.body.className = 'lang-' + lang;
        
        toggleBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === lang);
        });
        
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.dataset.i18n;
            if (translations[lang][key]) {
                el.textContent = translations[lang][key];
            }
        });
        
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.placeholder = translations[lang].search;
        }
    };

    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            applyLang(btn.dataset.lang);
        });
    });

    applyLang('en');
}

// ═══════════════════════════════════════════════════════════
// INTEGRATED SEARCH FILTERING
// ═══════════════════════════════════════════════════════════
function initSearch() {
    const searchInput = document.querySelector('.search-input');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        state.searchQuery = query;

        const links = document.querySelectorAll('.nav-link');
        const sections = document.querySelectorAll('.doc-section');

        if (!query) {
            links.forEach(l => l.style.display = '');
            sections.forEach(s => {
                s.querySelectorAll('.card, tr, .callout-card').forEach(el => {
                    el.style.display = '';
                    el.style.border = '';
                });
            });
            return;
        }

        links.forEach(link => {
            const targetId = link.getAttribute('href');
            const targetSec = document.querySelector(targetId);
            if (!targetSec) return;

            const contentText = targetSec.innerText.toLowerCase();
            const linkText = link.innerText.toLowerCase();

            if (contentText.includes(query) || linkText.includes(query)) {
                link.style.display = '';
            } else {
                link.style.display = 'none';
            }
        });

        const activeSection = document.querySelector(state.currentHash);
        if (activeSection) {
            activeSection.querySelectorAll('.card, tr, .callout-card').forEach(item => {
                const text = item.innerText.toLowerCase();
                if (text.includes(query)) {
                    item.style.display = '';
                    item.style.border = '1px solid rgba(0, 240, 255, 0.4)';
                } else {
                    item.style.display = 'none';
                }
            });
        }
    });
}

// ═══════════════════════════════════════════════════════════
// CODE BLOCKS COPY HELPER
// ═══════════════════════════════════════════════════════════
function initCodeBlocks() {
    document.querySelectorAll('pre').forEach(pre => {
        const wrapper = document.createElement('div');
        wrapper.className = 'code-container';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';
        pre.parentNode.insertBefore(btn, pre);

        btn.addEventListener('click', () => {
            const code = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
            navigator.clipboard.writeText(code).then(() => {
                btn.textContent = 'Copied!';
                btn.style.background = '#00ffaa';
                btn.style.color = '#000';
                
                showToast(translations[state.lang].copied);
                
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.style.background = '';
                    btn.style.color = '';
                }, 2000);
            });
        });
    });
}

function showToast(msg) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        document.body.appendChild(toast);
    }
    toast.innerHTML = `<span class="pulse-dot"></span> ${msg}`;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2500);
}

// ═══════════════════════════════════════════════════════════
// INTERACTIVE WIDGETS / SIMULATORS
// ═══════════════════════════════════════════════════════════
function initDemos() {
    initRaytracerDemo();
    initWaveDemo();
    initNodeGraphDemo();
    initFalloffCurveDemo();
}

/* 1. Canvas Dynamic Adaptable Path-Tracer */
function initRaytracerDemo() {
    const canvas = document.getElementById('raytracer-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const scale = 2; // low-res progressive upscale
    let w = Math.floor(canvas.clientWidth / scale);
    let h = Math.floor(canvas.clientHeight / scale);
    canvas.width = w;
    canvas.height = h;

    let accumBuffer = new Float32Array(w * h * 3);
    let samples = 0;
    let cameraPos = { x: 0, y: 0.15, z: 2.2 };
    let lightPos = { x: 1.2, y: 1.5, z: 0.8 };
    let isMoving = false;
    let isMouseDown = false;
    let time = 0;

    function dot(x1, y1, z1, x2, y2, z2) {
        return x1 * x2 + y1 * y2 + z1 * z2;
    }

    function intersectSphere(ox, oy, oz, dx, dy, dz, sx, sy, sz, sr) {
        let ocx = ox - sx;
        let ocy = oy - sy;
        let ocz = oz - sz;
        let b = ocx * dx + ocy * dy + ocz * dz;
        let c = ocx * ocx + ocy * ocy + ocz * ocz - sr * sr;
        let disc = b * b - c;
        if (disc < 0) return -1;
        let t = -b - Math.sqrt(disc);
        return t > 0.001 ? t : -1;
    }

    function trace(ox, oy, oz, dx, dy, dz, depth) {
        if (depth > 2) return [0, 0, 0];

        let tNear = 1e20;
        let hitType = 0; // 0: none, 1: floor, 2: center object (sculpt/paint/sphere), 3: gold sphere, 4: blue sphere
        let hx = 0, hy = 0, hz = 0;
        let nx = 0, ny = 0, nz = 0;

        const mode = state.viewportMode;

        // Spheres layout depending on viewport mode
        let sCenterRadius = 0.42;
        let centerObjectZ = -0.9;
        
        // CENTER SPHERE (Active sculpt/paint object)
        let s1x = 0.0, s1y = -0.08, s1z = centerObjectZ;
        
        // Center sphere ray intersection (Handles dynamic displacement for sculpting!)
        if (mode === 'sculpt') {
            // Sculpting: Ray check with rough bounding box first
            let tBound = intersectSphere(ox, oy, oz, dx, dy, dz, s1x, s1y, s1z, sCenterRadius + 0.15);
            if (tBound > 0) {
                // Raymarch to find displacement intersection
                let step = 0.01;
                let currentT = tBound;
                let hitDisplaced = false;
                for (let stepIdx = 0; stepIdx < 30; stepIdx++) {
                    let rx = ox + dx * currentT;
                    let ry = oy + dy * currentT;
                    let rz = oz + dz * currentT;
                    
                    let dx_c = rx - s1x;
                    let dy_c = ry - s1y;
                    let dz_c = rz - s1z;
                    let dist = Math.sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c);
                    
                    // Map to UV for displacement
                    let normalX = dx_c / dist;
                    let normalY = dy_c / dist;
                    let normalZ = dz_c / dist;
                    let u = Math.floor((0.5 + Math.atan2(normalZ, normalX) / (2 * Math.PI)) * (paintTextureWidth - 1));
                    let v = Math.floor((0.5 - Math.asin(normalY) / Math.PI) * (paintTextureHeight - 1));
                    
                    u = Math.max(0, Math.min(paintTextureWidth - 1, u));
                    v = Math.max(0, Math.min(paintTextureHeight - 1, v));
                    let displacement = sculptDisplacement[v * paintTextureWidth + u];
                    
                    let radiusDisplaced = sCenterRadius + displacement;
                    if (dist <= radiusDisplaced) {
                        tNear = currentT;
                        hitType = 2;
                        hx = rx; hy = ry; hz = rz;
                        nx = normalX; ny = normalY; nz = normalZ;
                        hitDisplaced = true;
                        break;
                    }
                    currentT += step;
                }
            }
        } else if (mode !== 'world' && mode !== 'water' && mode !== 'nodes' && mode !== 'terrain') {
            // Standard center sphere
            let t1 = intersectSphere(ox, oy, oz, dx, dy, dz, s1x, s1y, s1z, sCenterRadius);
            if (t1 > 0 && t1 < tNear) {
                tNear = t1;
                hitType = 2;
                hx = ox + dx * tNear;
                hy = oy + dy * tNear;
                hz = oz + dz * tNear;
                nx = (hx - s1x) / sCenterRadius;
                ny = (hy - s1y) / sCenterRadius;
                nz = (hz - s1z) / sCenterRadius;
            }
        }

        // Side Spheres (only visible in Home, Render, Camera, System)
        const showSides = (mode === 'home' || mode === 'render' || mode === 'camera' || mode === 'system');
        
        let s2x = 0.75, s2y = -0.15, s2z = -1.2, s2r = 0.35;
        if (showSides) {
            let t2 = intersectSphere(ox, oy, oz, dx, dy, dz, s2x, s2y, s2z, s2r);
            if (t2 > 0 && t2 < tNear) {
                tNear = t2;
                hitType = 3;
                hx = ox + dx * tNear;
                hy = oy + dy * tNear;
                hz = oz + dz * tNear;
                nx = (hx - s2x) / s2r;
                ny = (hy - s2y) / s2r;
                nz = (hz - s2z) / s2r;
            }
        }

        let s3x = -0.75, s3y = -0.15, s3z = -1.2, s3r = 0.35;
        if (showSides) {
            let t3 = intersectSphere(ox, oy, oz, dx, dy, dz, s3x, s3y, s3z, s3r);
            if (t3 > 0 && t3 < tNear) {
                tNear = t3;
                hitType = 4;
                hx = ox + dx * tNear;
                hy = oy + dy * tNear;
                hz = oz + dz * tNear;
                nx = (hx - s3x) / s3r;
                ny = (hy - s3y) / s3r;
                nz = (hz - s3z) / s3r;
            }
        }

        // Floor / Terrain Wavy plane
        let tFloor = 1e20;
        let isFloorHit = false;
        
        if (mode === 'water') {
            // Interactive path-traced WAVY ocean surface grid
            let waveT = 1e20;
            // Raymarch displacement plane
            let currentT = 0.5;
            for (let stepIdx = 0; stepIdx < 40; stepIdx++) {
                let rx = ox + dx * currentT;
                let ry = oy + dy * currentT;
                let rz = oz + dz * currentT;
                
                // Gerstner/Sine Wave deformation
                let height = -0.4 + 0.08 * Math.sin(rx * 5.0 + time * 1.5) * Math.cos(rz * 5.0 + time * 1.2);
                if (ry <= height) {
                    waveT = currentT;
                    hx = rx; hy = ry; hz = rz;
                    // Finite differences normal estimation
                    let eps = 0.01;
                    let hLeft = -0.4 + 0.08 * Math.sin((rx - eps) * 5.0 + time * 1.5) * Math.cos(rz * 5.0 + time * 1.2);
                    let hRight = -0.4 + 0.08 * Math.sin((rx + eps) * 5.0 + time * 1.5) * Math.cos(rz * 5.0 + time * 1.2);
                    let hBack = -0.4 + 0.08 * Math.sin(rx * 5.0 + time * 1.5) * Math.cos((rz - eps) * 5.0 + time * 1.2);
                    let hFront = -0.4 + 0.08 * Math.sin(rx * 5.0 + time * 1.5) * Math.cos((rz + eps) * 5.0 + time * 1.2);
                    
                    nx = (hLeft - hRight);
                    ny = eps * 2.0;
                    nz = (hBack - hFront);
                    let lenN = Math.sqrt(nx*nx + ny*ny + nz*nz);
                    nx /= lenN; ny /= lenN; nz /= lenN;
                    
                    isFloorHit = true;
                    break;
                }
                currentT += 0.04;
            }
            if (isFloorHit && waveT < tNear) {
                tNear = waveT;
                hitType = 1;
            } else {
                isFloorHit = false;
            }
        } else if (mode === 'nodes' || mode === 'terrain') {
            // Procedural mountain displacement
            let mountainT = 1e20;
            let currentT = 0.5;
            for (let stepIdx = 0; stepIdx < 45; stepIdx++) {
                let rx = ox + dx * currentT;
                let ry = oy + dy * currentT;
                let rz = oz + dz * currentT;
                
                // Fractal Perlin/Sine noise
                let hNoise = -0.55 + 0.15 * Math.sin(rx * 2.4) * Math.sin(rz * 2.1) + 0.05 * Math.sin(rx * 8.0) * Math.cos(rz * 6.5);
                if (ry <= hNoise) {
                    mountainT = currentT;
                    hx = rx; hy = ry; hz = rz;
                    // Normal estimation
                    let eps = 0.01;
                    let hl = -0.55 + 0.15 * Math.sin((rx - eps) * 2.4) * Math.sin(rz * 2.1) + 0.05 * Math.sin((rx - eps) * 8.0) * Math.cos(rz * 6.5);
                    let hr = -0.55 + 0.15 * Math.sin((rx + eps) * 2.4) * Math.sin(rz * 2.1) + 0.05 * Math.sin((rx + eps) * 8.0) * Math.cos(rz * 6.5);
                    let hb = -0.55 + 0.15 * Math.sin(rx * 2.4) * Math.sin((rz - eps) * 2.1) + 0.05 * Math.sin(rx * 8.0) * Math.cos((rz - eps) * 6.5);
                    let hf = -0.55 + 0.15 * Math.sin(rx * 2.4) * Math.sin((rz + eps) * 2.1) + 0.05 * Math.sin(rx * 8.0) * Math.cos((rz + eps) * 6.5);
                    
                    nx = (hl - hr);
                    ny = eps * 2.0;
                    nz = (hb - hf);
                    let lenN = Math.sqrt(nx*nx + ny*ny + nz*nz);
                    nx /= lenN; ny /= lenN; nz /= lenN;
                    
                    isFloorHit = true;
                    break;
                }
                currentT += 0.04;
            }
            if (isFloorHit && mountainT < tNear) {
                tNear = mountainT;
                hitType = 1;
            } else {
                isFloorHit = false;
            }
        } else {
            // Standard Checkers Floor
            tFloor = (-0.5 - oy) / dy;
            if (tFloor > 0.001 && tFloor < tNear) {
                tNear = tFloor;
                hitType = 1;
                hx = ox + dx * tNear;
                hy = -0.5;
                hz = oz + dz * tNear;
                nx = 0; ny = 1; nz = 0;
            }
        }

        if (hitType === 0) {
            // Sky gradient (Nishita Style)
            let t = 0.5 * (dy + 1.0);
            let skyR = (1.0 - t) * 0.08 + t * 0.55;
            let skyG = (1.0 - t) * 0.12 + t * 0.72;
            let skyB = (1.0 - t) * 0.20 + t * 0.95;
            
            // Atmospheric fog overlay in world mode
            if (mode === 'world') {
                // Foggy density haze
                skyR += 0.08; skyG += 0.08; skyB += 0.08;
            }
            return [skyR, skyG, skyB];
        }

        // Direction to light source
        let lx = lightPos.x - hx;
        let ly = lightPos.y - hy;
        let lz = lightPos.z - hz;
        let lenL = Math.sqrt(lx*lx + ly*ly + lz*lz);
        lx /= lenL; ly /= lenL; lz /= lenL;

        // Shadow offset
        let shx = hx + nx * 0.002;
        let shy = hy + ny * 0.002;
        let shz = hz + nz * 0.002;

        let inShadow = false;
        
        // Bounding box collision for shadows
        if (mode !== 'world' && mode !== 'water' && mode !== 'nodes' && mode !== 'terrain') {
            if (intersectSphere(shx, shy, shz, lx, ly, lz, s1x, s1y, s1z, sCenterRadius) > 0.001) {
                inShadow = true;
            }
        }
        if (showSides) {
            if (intersectSphere(shx, shy, shz, lx, ly, lz, s2x, s2y, s2z, s2r) > 0.001 ||
                intersectSphere(shx, shy, shz, lx, ly, lz, s3x, s3y, s3z, s3r) > 0.001) {
                inShadow = true;
            }
        }

        let diff = dot(nx, ny, nz, lx, ly, lz);
        if (diff < 0) diff = 0;
        let sh = inShadow ? 0.25 : 0.95;

        // BASE COLORS DEFINITION
        let r = 0.8, g = 0.8, b = 0.8;
        
        if (hitType === 1) { // Checkers or terrain Splatmap
            if (mode === 'water') {
                // Ocean color (dark blue, highly reflective)
                r = 0.05; g = 0.16; b = 0.24;
            } else if (mode === 'nodes' || mode === 'terrain') {
                // Splat map (Grass green at bottom, Rock brown on slopes)
                let slope = 1.0 - ny;
                if (slope > 0.45) {
                    r = 0.22; g = 0.18; b = 0.15; // Rock
                } else {
                    r = 0.12; g = 0.28; b = 0.08; // Grass
                }
            } else {
                let check = (Math.floor(hx * 2.2) + Math.floor(hz * 2.2)) % 2 === 0;
                r = check ? 0.08 : 0.18;
                g = check ? 0.09 : 0.20;
                b = check ? 0.12 : 0.26;
            }
        } else if (hitType === 2) { 
            // Center Sphere - Special materials per subpage
            if (mode === 'mesh-paint') {
                // Read from paintTexture using hit UV coordinates
                let dx_c = hx - s1x;
                let dy_c = hy - s1y;
                let dz_c = hz - s1z;
                let dist = Math.sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c);
                let normalX = dx_c / dist;
                let normalY = dy_c / dist;
                let normalZ = dz_c / dist;
                
                let u = Math.floor((0.5 + Math.atan2(normalZ, normalX) / (2 * Math.PI)) * (paintTextureWidth - 1));
                let v = Math.floor((0.5 - Math.asin(normalY) / Math.PI) * (paintTextureHeight - 1));
                
                u = Math.max(0, Math.min(paintTextureWidth - 1, u));
                v = Math.max(0, Math.min(paintTextureHeight - 1, v));
                
                const pIdx = (v * paintTextureWidth + u) * 3;
                r = paintTexture[pIdx] / 255;
                g = paintTexture[pIdx+1] / 255;
                b = paintTexture[pIdx+2] / 255;
            } else if (mode === 'sculpt') {
                r = 0.45; g = 0.48; b = 0.52; // Sculpt clay grey
            } else if (mode === 'hair') {
                r = 0.22; g = 0.15; b = 0.11; // Hair base brown
            } else {
                // Default red center
                r = 0.85; g = 0.15; b = 0.15;
            }
        } else if (hitType === 3) { // Gold sphere
            r = 0.95; g = 0.82; b = 0.25;
            let ddn = dot(dx, dy, dz, nx, ny, nz);
            let rx = dx - 2 * ddn * nx;
            let ry = dy - 2 * ddn * ny;
            let rz = dz - 2 * ddn * nz;
            
            let f = 0.03;
            rx += (Math.random() - 0.5) * f;
            ry += (Math.random() - 0.5) * f;
            rz += (Math.random() - 0.5) * f;
            let lenR = Math.sqrt(rx*rx + ry*ry + rz*rz);
            rx /= lenR; ry /= lenR; rz /= lenR;

            let bounce = trace(shx, shy, shz, rx, ry, rz, depth + 1);
            return [
                (r * diff * 0.1 + bounce[0] * 0.9) * sh,
                (g * diff * 0.1 + bounce[1] * 0.9) * sh,
                (b * diff * 0.1 + bounce[2] * 0.9) * sh
            ];
        } else if (hitType === 4) { // Blue sphere
            r = 0.15; g = 0.55; b = 0.95;
        }

        // Reflection for Wavy Water surface
        if (mode === 'water' && hitType === 1) {
            let ddn = dot(dx, dy, dz, nx, ny, nz);
            let rx = dx - 2 * ddn * nx;
            let ry = dy - 2 * ddn * ny;
            let rz = dz - 2 * ddn * nz;
            let lenR = Math.sqrt(rx*rx + ry*ry + rz*rz);
            rx /= lenR; ry /= lenR; rz /= lenR;
            
            let skyRefl = trace(shx, shy, shz, rx, ry, rz, depth + 1);
            // Blend reflection with deep water body
            return [
                (0.12 * diff + skyRefl[0] * 0.88) * sh,
                (0.24 * diff + skyRefl[1] * 0.88) * sh,
                (0.38 * diff + skyRefl[2] * 0.88) * sh
            ];
        }

        // Lambertian bounce
        let rx = nx + (Math.random() * 2.0 - 1.0);
        let ry = ny + (Math.random() * 2.0 - 1.0);
        let rz = nz + (Math.random() * 2.0 - 1.0);
        let lenR = Math.sqrt(rx*rx + ry*ry + rz*rz);
        if (lenR > 0.001) { rx /= lenR; ry /= lenR; rz /= lenR; }
        else { rx = nx; ry = ny; rz = nz; }

        let bounce = trace(shx, shy, shz, rx, ry, rz, depth + 1);

        // Volumetric distance mist absorption in World mode
        let absorption = 1.0;
        if (mode === 'world') {
            absorption = Math.exp(-tNear * 0.15);
        }

        let finalR = (r * diff * 0.65 + bounce[0] * 0.35 * r) * sh * absorption;
        let finalG = (g * diff * 0.65 + bounce[1] * 0.35 * g) * sh * absorption;
        let finalB = (b * diff * 0.65 + bounce[2] * 0.35 * b) * sh * absorption;
        
        // Add local fog scattering
        if (mode === 'world') {
            let fogFactor = 1.0 - absorption;
            finalR += fogFactor * 0.55;
            finalG += fogFactor * 0.62;
            finalB += fogFactor * 0.72;
        }

        return [finalR, finalG, finalB];
    }

    // Proj 3D point to 2D screen coordinate for skeletal and wireframe overlays
    function project(x, y, z) {
        // Simple perspective projection mock
        let fov = 45 * Math.PI / 180;
        let aspect = w / h;
        // Transform based on cameraPos (camera looks down Z at -1)
        let camX = x - cameraPos.x;
        let camY = y - cameraPos.y;
        let camZ = z - cameraPos.z;
        
        let screenX = w / 2 + (camX / -camZ) * (w / (2 * Math.tan(fov/2)));
        let screenY = h / 2 - (camY / -camZ) * (h / (2 * Math.tan(fov/2) * aspect));
        return { x: screenX, y: screenY, visible: camZ < 0 };
    }

    function renderFrame() {
        const mode = state.viewportMode;
        
        // 1. Clean drawing and path tracing
        const imgData = ctx.createImageData(w, h);
        let passes = isMoving ? 1 : 2;
        
        if (isMoving) {
            samples = 0;
            accumBuffer.fill(0);
        }
        
        samples += passes;
        time += 0.02;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 3;
                let r = 0, g = 0, b = 0;

                for (let s = 0; s < passes; s++) {
                    const u = (x + Math.random()) / w;
                    const v = (y + Math.random()) / h;
                    
                    const fov = 45 * Math.PI / 180;
                    const aspect = w / h;
                    const cx = (2.0 * u - 1.0) * Math.tan(fov/2) * aspect;
                    const cy = (1.0 - 2.0 * v) * Math.tan(fov/2);
                    
                    let dx = cx, dy = cy, dz = -1;
                    let lenD = Math.sqrt(dx*dx + dy*dy + dz*dz);
                    dx /= lenD; dy /= lenD; dz /= lenD;

                    const c = trace(cameraPos.x, cameraPos.y, cameraPos.z, dx, dy, dz, 0);
                    r += c[0];
                    g += c[1];
                    b += c[2];
                }

                accumBuffer[idx] += r;
                accumBuffer[idx+1] += g;
                accumBuffer[idx+2] += b;

                let finalR = Math.sqrt(accumBuffer[idx] / samples);
                let finalG = Math.sqrt(accumBuffer[idx+1] / samples);
                let finalB = Math.sqrt(accumBuffer[idx+2] / samples);

                // Vulkan RT / OptiX comparison split screen simulation
                if (mode === 'render') {
                    if (x < w / 2) {
                        // Vulkan RT - simulate noise and speckles (fireflies)
                        let noiseIntensity = Math.max(0, 1.0 - samples * 0.08);
                        if (Math.random() < noiseIntensity * 0.15) {
                            finalR = Math.random() > 0.8 ? 1.4 : finalR + 0.2;
                            finalG = finalG + 0.1;
                            finalB = finalB + 0.2;
                        }
                    }
                }

                const pixelIdx = (y * w + x) * 4;
                imgData.data[pixelIdx] = Math.min(255, Math.floor(finalR * 255));
                imgData.data[pixelIdx+1] = Math.min(255, Math.floor(finalG * 255));
                imgData.data[pixelIdx+2] = Math.min(255, Math.floor(finalB * 255));
                imgData.data[pixelIdx+3] = 255;
            }
        }
        
        ctx.putImageData(imgData, 0, 0);

        // 2. VIEWPORT TEXT & HUD OVERLAYS (SKELETAL BONES, VIEWFINDFER HUD)
        const HUDCanvas = document.createElement('canvas');
        HUDCanvas.width = canvas.width;
        HUDCanvas.height = canvas.height;
        const hCtx = HUDCanvas.getContext('2d');
        
        // Draw split screen boundary bar in Render Mode
        if (mode === 'render') {
            hCtx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
            hCtx.lineWidth = 2;
            hCtx.beginPath();
            hCtx.moveTo(w / 2, 0);
            hCtx.lineTo(w / 2, h);
            hCtx.stroke();
            
            // Labels
            hCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            hCtx.font = 'bold 8px Outfit';
            hCtx.fillText("VULKAN RT (Noisy)", 10, h - 10);
            hCtx.fillText("OptiX OIDN (Clean)", w / 2 + 10, h - 10);
        }

        // Draw Skeletal bone hierarchy in Animation Mode
        if (mode === 'animation') {
            hCtx.strokeStyle = '#00ffaa';
            hCtx.lineWidth = 1.5;
            hCtx.fillStyle = '#ffb700';

            // Walk animation joints coordinates
            let cycle = time * 2.0;
            let hipY = -0.15;
            let hip = project(0.0, hipY, -0.9);
            let spine = project(0.0, hipY + 0.25, -0.9);
            let head = project(0.0, hipY + 0.38, -0.9);
            
            let kneeL_y = hipY - 0.16 + 0.05 * Math.sin(cycle);
            let kneeL_x = 0.08 * Math.cos(cycle);
            let footL_y = hipY - 0.32 + 0.02 * Math.sin(cycle + 1);
            let footL_x = 0.14 * Math.cos(cycle);

            let kneeR_y = hipY - 0.16 + 0.05 * Math.sin(cycle + Math.PI);
            let kneeR_x = 0.08 * Math.cos(cycle + Math.PI);
            let footR_y = hipY - 0.32 + 0.02 * Math.sin(cycle + Math.PI + 1);
            let footR_x = 0.14 * Math.cos(cycle + Math.PI);

            let kneeL = project(kneeL_x, kneeL_y, -0.9);
            let footL = project(footL_x, footL_y, -0.9);
            let kneeR = project(kneeR_x, kneeR_y, -0.9);
            let footR = project(footR_x, footR_y, -0.9);

            const drawBone = (p1, p2) => {
                if (p1.visible && p2.visible) {
                    hCtx.beginPath();
                    hCtx.moveTo(p1.x, p1.y);
                    hCtx.lineTo(p2.x, p2.y);
                    hCtx.stroke();
                }
            };
            const drawJoint = (p) => {
                if (p.visible) {
                    hCtx.beginPath();
                    hCtx.arc(p.x, p.y, 2, 0, Math.PI*2);
                    hCtx.fill();
                }
            };

            // Draw skeleton lines
            drawBone(hip, spine);
            drawBone(spine, head);
            drawBone(hip, kneeL);
            drawBone(kneeL, footL);
            drawBone(hip, kneeR);
            drawBone(kneeR, footR);

            drawJoint(hip);
            drawJoint(spine);
            drawJoint(head);
            drawJoint(kneeL);
            drawJoint(footL);
            drawJoint(kneeR);
            drawJoint(footR);
        }

        // Draw non-destructive wireframe topology edit mode
        if (mode === 'edit-mesh') {
            hCtx.strokeStyle = 'rgba(255, 183, 0, 0.7)';
            hCtx.lineWidth = 1;
            hCtx.fillStyle = '#00f0ff';
            
            // Vertex corners of 3D box
            let boxSize = 0.25;
            let vertices = [
                {x: -boxSize, y: -boxSize, z: -0.9 - boxSize},
                {x: boxSize, y: -boxSize, z: -0.9 - boxSize},
                {x: boxSize, y: boxSize, z: -0.9 - boxSize},
                {x: -boxSize, y: boxSize, z: -0.9 - boxSize},
                {x: -boxSize, y: -boxSize, z: -0.9 + boxSize},
                {x: boxSize, y: -boxSize, z: -0.9 + boxSize},
                {x: boxSize, y: boxSize, z: -0.9 + boxSize},
                {x: -boxSize, y: boxSize, z: -0.9 + boxSize}
            ];

            let proj = vertices.map(v => project(v.x, v.y, v.z));
            
            // Connect cube edges
            const drawEdge = (i1, i2) => {
                if (proj[i1].visible && proj[i2].visible) {
                    hCtx.beginPath();
                    hCtx.moveTo(proj[i1].x, proj[i1].y);
                    hCtx.lineTo(proj[i2].x, proj[i2].y);
                    hCtx.stroke();
                }
            };

            // Draw front face edges
            drawEdge(0, 1); drawEdge(1, 2); drawEdge(2, 3); drawEdge(3, 0);
            // Draw back face edges
            drawEdge(4, 5); drawEdge(5, 6); drawEdge(6, 7); drawEdge(7, 4);
            // Draw columns edges
            drawEdge(0, 4); drawEdge(1, 5); drawEdge(2, 6); drawEdge(3, 7);

            // Draw vertices as glowing dots
            proj.forEach(p => {
                if (p.visible) {
                    hCtx.beginPath();
                    hCtx.arc(p.x, p.y, 2.5, 0, Math.PI*2);
                    hCtx.fill();
                }
            });
        }

        // Draw Foliage trees cone outlines in Terrain / Erosion workspace
        if (mode === 'terrain') {
            hCtx.fillStyle = 'rgba(0, 255, 170, 0.4)';
            hCtx.strokeStyle = 'rgba(0, 255, 170, 0.8)';
            
            let trees = [
                {x: -0.5, z: -1.0},
                {x: 0.2, z: -1.1},
                {x: -0.2, z: -1.3},
                {x: 0.6, z: -0.9}
            ];

            trees.forEach(t => {
                let rx = t.x;
                let rz = t.z;
                let ry = -0.55 + 0.15 * Math.sin(rx * 2.4) * Math.sin(rz * 2.1) + 0.05 * Math.sin(rx * 8.0) * Math.cos(rz * 6.5);
                
                let base = project(rx, ry, rz);
                let tip = project(rx, ry + 0.2, rz);

                if (base.visible && tip.visible) {
                    hCtx.beginPath();
                    hCtx.moveTo(base.x - 4, base.y);
                    hCtx.lineTo(base.x + 4, base.y);
                    hCtx.lineTo(tip.x, tip.y);
                    hCtx.closePath();
                    hCtx.fill();
                    hCtx.stroke();
                }
            });
        }

        // Draw growing groom hair curves
        if (mode === 'hair') {
            hCtx.strokeStyle = 'rgba(255, 120, 40, 0.55)';
            hCtx.lineWidth = 1;
            
            // Draw 60 short curved lines radiating from center sphere
            let numStrands = 50;
            let s1x = 0.0, s1y = -0.08, s1z = centerObjectZ;
            for (let i = 0; i < numStrands; i++) {
                let theta = (i / numStrands) * Math.PI * 2 + time * 0.2;
                let phi = (Math.sin(i * 123) * 0.5 + 0.5) * Math.PI;

                let nx = Math.sin(phi) * Math.cos(theta);
                let ny = Math.cos(phi);
                let nz = Math.sin(phi) * Math.sin(theta);
                
                let p1 = project(s1x + nx * sCenterRadius, s1y + ny * sCenterRadius, s1z + nz * sCenterRadius);
                let p2 = project(s1x + nx * (sCenterRadius + 0.14), s1y + ny * (sCenterRadius + 0.14) - 0.06, s1z + nz * (sCenterRadius + 0.14));

                if (p1.visible && p2.visible) {
                    hCtx.beginPath();
                    hCtx.moveTo(p1.x, p1.y);
                    hCtx.quadraticCurveTo(p1.x + nx * 5, p1.y + ny * 5, p2.x, p2.y);
                    hCtx.stroke();
                }
            }
        }

        // Draw system log lines directly on the console screen
        if (mode === 'system') {
            hCtx.fillStyle = 'rgba(0, 0, 0, 0.65)';
            hCtx.fillRect(10, 10, w - 20, 55);
            
            hCtx.fillStyle = '#00ffaa';
            hCtx.font = '6px Fira Code, Courier New';
            let line1 = `[RayTrophi Studio] Loading configuration profile...`;
            let line2 = `[CUDA] OptiX context initialized on Device 0. (Active: OptiX 7.5)`;
            let line3 = `[AssetManager] Loaded skeleton structure (bones: 98, vertices: 142K)`;
            let line4 = `[Info] viewport convergence reached. rendering final accumulation.`;

            let scrollOffset = Math.floor(time * 8.0) % 4;
            hCtx.fillText(line1, 14, 20);
            hCtx.fillText(line2, 14, 30);
            hCtx.fillText(line3, 14, 40);
            hCtx.fillText(line4, 14, 50);
        }

        // Draw Viewfinder diagnostics in Camera Mode (Sobel edge colors peaking, Zebra overexposure stripes, Histogram)
        if (mode === 'camera') {
            // Read canvas pixel data to run Sobel Focus Peaking filter
            const frameData = ctx.getImageData(0, 0, w, h);
            const data = frameData.data;

            // Simple Focus Peaking edge outline
            hCtx.strokeStyle = 'rgba(0, 255, 170, 0.9)'; // Green highlights
            hCtx.lineWidth = 1;
            hCtx.beginPath();

            for (let y = 1; y < h - 1; y += 2) {
                for (let x = 1; x < w - 1; x += 2) {
                    let idx = (y * w + x) * 4;
                    let luma = 0.299 * data[idx] + 0.587 * data[idx+1] + 0.114 * data[idx+2];
                    
                    let idxRight = (y * w + (x + 1)) * 4;
                    let lumaRight = 0.299 * data[idxRight] + 0.587 * data[idxRight+1] + 0.114 * data[idxRight+2];

                    let idxDown = ((y + 1) * w + x) * 4;
                    let lumaDown = 0.299 * data[idxDown] + 0.587 * data[idxDown+1] + 0.114 * data[idxDown+2];

                    let dx_val = Math.abs(luma - lumaRight);
                    let dy_val = Math.abs(luma - lumaDown);
                    
                    // Edge detected
                    if (dx_val + dy_val > 25) {
                        hCtx.fillRect(x, y, 1, 1);
                    }

                    // Zebra Stripes overexposure (for pixels where brightness > 220)
                    if (luma > 220) {
                        // Drawing moving diagonal zebra stripes
                        let stripeTime = Math.floor(time * 15) % 10;
                        if ((x + y + stripeTime) % 8 === 0) {
                            hCtx.fillStyle = 'rgba(0, 0, 0, 0.95)';
                            hCtx.fillRect(x, y, 1.2, 1.2);
                        }
                    }
                }
            }

            // Draw Center Autofocus Area brackets
            let cSize = 16;
            hCtx.strokeStyle = '#00ffaa';
            hCtx.lineWidth = 1.2;
            hCtx.beginPath();
            // Top Left corner bracket
            hCtx.moveTo(w / 2 - cSize, h / 2 - cSize / 2);
            hCtx.lineTo(w / 2 - cSize, h / 2 - cSize);
            hCtx.lineTo(w / 2 - cSize / 2, h / 2 - cSize);
            // Top Right bracket
            hCtx.moveTo(w / 2 + cSize / 2, h / 2 - cSize);
            hCtx.lineTo(w / 2 + cSize, h / 2 - cSize);
            hCtx.lineTo(w / 2 + cSize, h / 2 - cSize / 2);
            // Bottom Left
            hCtx.moveTo(w / 2 - cSize, h / 2 + cSize / 2);
            hCtx.lineTo(w / 2 - cSize, h / 2 + cSize);
            hCtx.lineTo(w / 2 - cSize / 2, h / 2 + cSize);
            // Bottom Right
            hCtx.moveTo(w / 2 + cSize / 2, h / 2 + cSize);
            hCtx.lineTo(w / 2 + cSize, h / 2 + cSize);
            hCtx.lineTo(w / 2 + cSize, h / 2 + cSize / 2);
            hCtx.stroke();

            // Real-time RGB/Luma Histogram overlay in top right corner
            let histW = 55;
            let histH = 30;
            let histX = w - histW - 10;
            let histY = 10;
            
            hCtx.fillStyle = 'rgba(11, 13, 22, 0.8)';
            hCtx.fillRect(histX, histY, histW, histH);
            hCtx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            hCtx.strokeRect(histX, histY, histW, histH);

            let bins = new Uint32Array(histW);
            for (let i = 0; i < data.length; i += 16) {
                let l = Math.floor((0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]) * (histW - 1) / 255);
                bins[l]++;
            }
            let maxCount = 1;
            for (let i = 0; i < histW; i++) {
                if (bins[i] > maxCount) maxCount = bins[i];
            }

            hCtx.fillStyle = '#00f0ff';
            for (let i = 0; i < histW; i++) {
                let barH = (bins[i] / maxCount) * (histH - 2);
                hCtx.fillRect(histX + i, histY + histH - barH - 1, 1, barH);
            }
        }

        // Draw overlays onto the main progressive path tracer canvas
        ctx.drawImage(HUDCanvas, 0, 0);

        const sampleEl = document.getElementById('accum-samples');
        if (sampleEl) {
            sampleEl.textContent = samples;
        }

        requestAnimationFrame(renderFrame);
    }

    // Interactive Brushing (Paint and Sculpt)
    function handleBrushStroke(e) {
        const mode = state.viewportMode;
        if (mode !== 'sculpt' && mode !== 'mesh-paint') return;

        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Map mouse screen coordinate back to low-res coordinates
        const lx = mouseX / scale;
        const ly = mouseY / scale;

        // Perform ray intersection at mouse coordinates to find sphere hit point
        const uCoord = lx / w;
        const vCoord = ly / h;

        const fov = 45 * Math.PI / 180;
        const aspect = w / h;
        const cx = (2.0 * uCoord - 1.0) * Math.tan(fov/2) * aspect;
        const cy = (1.0 - 2.0 * vCoord) * Math.tan(fov/2);
        
        let dx = cx, dy = cy, dz = -1;
        let lenD = Math.sqrt(dx*dx + dy*dy + dz*dz);
        dx /= lenD; dy /= lenD; dz /= lenD;

        let s1x = 0.0, s1y = -0.08, s1z = centerObjectZ;
        let tHit = intersectSphere(cameraPos.x, cameraPos.y, cameraPos.z, dx, dy, dz, s1x, s1y, s1z, sCenterRadius + 0.1);
        
        if (tHit > 0) {
            let hx = cameraPos.x + dx * tHit;
            let hy = cameraPos.y + dy * tHit;
            let hz = cameraPos.z + dz * tHit;

            // Map hit point to spherical UV coordinates
            let dx_c = hx - s1x;
            let dy_c = hy - s1y;
            let dz_c = hz - s1z;
            let dist = Math.sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c);
            let normalX = dx_c / dist;
            let normalY = dy_c / dist;
            let normalZ = dz_c / dist;

            let hitU = 0.5 + Math.atan2(normalZ, normalX) / (2 * Math.PI);
            let hitV = 0.5 - Math.asin(normalY) / Math.PI;

            // Convert to texture coordinate space
            let tx = Math.floor(hitU * paintTextureWidth);
            let ty = Math.floor(hitV * paintTextureHeight);

            // Apply brush deformation (sculpt height displacement / paint brush values)
            let brushRadius = 8;
            for (let dy_b = -brushRadius; dy_b <= brushRadius; dy_b++) {
                for (let dx_b = -brushRadius; dx_b <= brushRadius; dx_b++) {
                    let d2 = dx_b*dx_b + dy_b*dy_b;
                    if (d2 <= brushRadius*brushRadius) {
                        let px = (tx + dx_b + paintTextureWidth) % paintTextureWidth;
                        let py = (ty + dy_b + paintTextureHeight) % paintTextureHeight;
                        
                        let weight = Math.exp(-d2 / (brushRadius * brushRadius * 0.5));

                        if (mode === 'mesh-paint') {
                            const pIdx = (py * paintTextureWidth + px) * 3;
                            // Paint blending with active color (Cyan color brush!)
                            paintTexture[pIdx] = Math.floor(paintTexture[pIdx] * (1.0 - weight * 0.3) + 0 * weight * 0.3);
                            paintTexture[pIdx+1] = Math.floor(paintTexture[pIdx+1] * (1.0 - weight * 0.3) + 240 * weight * 0.3);
                            paintTexture[pIdx+2] = Math.floor(paintTexture[pIdx+2] * (1.0 - weight * 0.3) + 255 * weight * 0.3);
                        } else if (mode === 'sculpt') {
                            // Sculpt height displacement buildup
                            sculptDisplacement[py * paintTextureWidth + px] += weight * 0.005;
                        }
                    }
                }
            }

            // Force viewport accumulation reset
            isMoving = true;
            clearTimeout(canvas.moveTimeout);
            canvas.moveTimeout = setTimeout(() => { isMoving = false; }, 100);
        }
    }

    canvas.addEventListener('mousedown', (e) => {
        isMouseDown = true;
        handleBrushStroke(e);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isMouseDown) {
            handleBrushStroke(e);
        } else {
            // Light source movement coordinate translation
            const rect = canvas.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * 2.0 - 1.0;
            const my = 1.0 - ((e.clientY - rect.top) / rect.height) * 2.0;

            lightPos.x = mx * 2.5;
            lightPos.y = my * 2.0 + 1.0;
            
            isMoving = true;
            clearTimeout(canvas.moveTimeout);
            canvas.moveTimeout = setTimeout(() => {
                isMoving = false;
            }, 120);
        }
    });

    window.addEventListener('mouseup', () => {
        isMouseDown = false;
    });

    renderFrame();
}

/* 2. Wave customizer deforming waves */
function initWaveDemo() {
    const canvas = document.getElementById('wave-visualizer');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const amplitudeSlider = document.getElementById('wave-amplitude');
    const choppinessSlider = document.getElementById('wave-choppiness');
    const ampVal = document.getElementById('amp-val');
    const chopVal = document.getElementById('chop-val');

    let time = 0;

    function drawWave() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const w = canvas.width;
        const h = canvas.height;
        const rawAmp = parseFloat(amplitudeSlider.value);
        const rawChop = parseFloat(choppinessSlider.value);
        
        if (ampVal) ampVal.textContent = rawAmp.toFixed(2);
        if (chopVal) chopVal.textContent = rawChop.toFixed(2);

        const amp = rawAmp * 35;
        const chop = rawChop;
        
        time += 0.04;

        for (let layer = 0; layer < 3; layer++) {
            ctx.beginPath();
            ctx.strokeStyle = layer === 0 ? '#00f0ff' : (layer === 1 ? 'rgba(159, 0, 255, 0.4)' : 'rgba(0, 255, 170, 0.3)');
            ctx.lineWidth = layer === 0 ? 3 : 1.5;
            
            const speedMult = layer === 0 ? 1 : (layer === 1 ? 0.7 : 1.4);
            const freq = layer === 0 ? 0.015 : (layer === 1 ? 0.025 : 0.01);
            const layerAmp = layer === 0 ? amp : (layer === 1 ? amp*0.6 : amp*0.3);

            for (let i = 0; i < w; i += 2) {
                const phase = i * freq - time * speedMult;
                const waveHeight = Math.sin(phase);
                const dx = chop * 12 * Math.cos(phase);
                
                const x = i - dx;
                const y = (h / 2 + layer * 15) + waveHeight * layerAmp;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }

        requestAnimationFrame(drawWave);
    }

    drawWave();
}

/* 3. Falloff Curve Editor */
function initFalloffCurveDemo() {
    const canvas = document.getElementById('falloff-editor');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let lut = new Array(64).fill(0);
    const N = 64;

    for (let i = 0; i < N; ++i) {
        const t = i / (N - 1);
        lut[i] = t * t * (3.0 - 2.0 * t);
    }

    function redraw() {
        ctx.fillStyle = '#07090f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#1a1f2e';
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const x = (canvas.width * i) / 4;
            const y = (canvas.height * i) / 4;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }

        ctx.strokeStyle = '#ffb700';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
            const x = (canvas.width * i) / (N - 1);
            const y = canvas.height * (1.0 - lut[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    function updateCurve(e) {
        const rect = canvas.getBoundingClientRect();
        const mx = Math.max(0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
        const my = Math.max(0, Math.min(1.0, 1.0 - (e.clientY - rect.top) / rect.height));

        const bin = Math.min(N - 1, Math.max(0, Math.floor(mx * (N - 1) + 0.5)));
        
        for (let d = -2; d <= 2; d++) {
            const b = bin + d;
            if (b >= 0 && b < N) {
                const weight = 1.0 - Math.abs(d) / 3.0;
                lut[b] = lut[b] * (1.0 - weight) + my * weight;
            }
        }
        redraw();
    }

    let isDrawing = false;
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        updateCurve(e);
    });
    window.addEventListener('mousemove', (e) => {
        if (isDrawing) updateCurve(e);
    });
    window.addEventListener('mouseup', () => {
        isDrawing = false;
    });

    const btnSmooth = document.getElementById('preset-smooth');
    const btnLinear = document.getElementById('preset-linear');
    
    if (btnSmooth) {
        btnSmooth.addEventListener('click', () => {
            for (let i = 0; i < N; ++i) {
                const t = i / (N - 1);
                lut[i] = t * t * (3.0 - 2.0 * t);
            }
            redraw();
        });
    }

    if (btnLinear) {
        btnLinear.addEventListener('click', () => {
            for (let i = 0; i < N; ++i) {
                lut[i] = i / (N - 1);
            }
            redraw();
        });
    }

    redraw();
}

/* 4. Sürükle Bırak Düğüm Editörü */
function initNodeGraphDemo() {
    const container = document.getElementById('node-graph-playground');
    if (!container) return;

    const nodes = [
        { id: 'node-perlin', el: document.getElementById('node-perlin'), x: 20, y: 50 },
        { id: 'node-erosion', el: document.getElementById('node-erosion'), x: 200, y: 120 },
        { id: 'node-output', el: document.getElementById('node-output'), x: 380, y: 70 }
    ];

    let activeNode = null;
    let dragOffset = { x: 0, y: 0 };

    nodes.forEach(node => {
        if (node.el) {
            node.el.style.left = `${node.x}px`;
            node.el.style.top = `${node.y}px`;
            
            const header = node.el.querySelector('.node-header-bar');
            header.addEventListener('mousedown', (e) => {
                activeNode = node;
                nodes.forEach(n => {
                    if (n.el) n.el.style.zIndex = 10;
                });
                node.el.style.zIndex = 20;

                const rect = node.el.getBoundingClientRect();
                dragOffset.x = e.clientX - rect.left;
                dragOffset.y = e.clientY - rect.top;
                
                document.querySelectorAll('.node-element').forEach(n => n.classList.remove('active'));
                node.el.classList.add('active');
                e.preventDefault();
            });
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (activeNode && activeNode.el) {
            const containerRect = container.getBoundingClientRect();
            let newX = e.clientX - containerRect.left - dragOffset.x;
            let newY = e.clientY - containerRect.top - dragOffset.y;

            newX = Math.max(0, Math.min(newX, containerRect.width - activeNode.el.clientWidth));
            newY = Math.max(0, Math.min(newY, containerRect.height - activeNode.el.clientHeight));

            activeNode.x = newX;
            activeNode.y = newY;

            activeNode.el.style.left = `${newX}px`;
            activeNode.el.style.top = `${newY}px`;

            drawConnections();
        }
    });

    window.addEventListener('mouseup', () => {
        activeNode = null;
    });

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '1';
    container.insertBefore(canvas, container.firstChild);

    function drawConnections() {
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const links = [
            { from: 'node-perlin', to: 'node-erosion' },
            { from: 'node-erosion', to: 'node-output' }
        ];

        ctx.lineWidth = 3;
        ctx.strokeStyle = '#00ffaa';

        links.forEach(link => {
            const fromNode = nodes.find(n => n.id === link.from);
            const toNode = nodes.find(n => n.id === link.to);

            if (fromNode && toNode && fromNode.el && toNode.el) {
                const fromPin = fromNode.el.querySelector('.pin-output');
                const toPin = toNode.el.querySelector('.pin-input');

                const fromRect = fromPin.getBoundingClientRect();
                const toRect = toPin.getBoundingClientRect();

                const x1 = fromRect.left - rect.left + fromRect.width / 2;
                const y1 = fromRect.top - rect.top + fromRect.height / 2;
                const x2 = toRect.left - rect.left + toRect.width / 2;
                const y2 = toRect.top - rect.top + toRect.height / 2;

                const controlOffset = Math.abs(x2 - x1) * 0.5;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.bezierCurveTo(x1 + controlOffset, y1, x2 - controlOffset, y2, x2, y2);
                
                ctx.shadowBlur = 8;
                ctx.shadowColor = '#00ffaa';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }
        });
    }

    setTimeout(drawConnections, 100);
}
