#include "SplashScreen.h"
#include <SDL_image.h>
#include <iostream>
#include <chrono>

SplashScreen::~SplashScreen() {
    close();
}

bool SplashScreen::init(const std::string& logoPath, int maxWidth, int maxHeight) {
    // Load logo image to get dimensions
    SDL_Surface* logoSurface = IMG_Load(logoPath.c_str());
    if (!logoSurface) {
        std::cerr << "[SplashScreen] Failed to load logo: " << logoPath << " - " << IMG_GetError() << std::endl;
        return false;
    }
    
    m_logoWidth = logoSurface->w;
    m_logoHeight = logoSurface->h;
    
    // Scale if too large
    float scale = 1.0f;
    if (m_logoWidth > maxWidth) {
        scale = (float)maxWidth / m_logoWidth;
    }
    if (m_logoHeight * scale > maxHeight) {
        scale = (float)maxHeight / m_logoHeight;
    }
    
    m_windowWidth = (int)(m_logoWidth * scale);
    m_windowHeight = (int)(m_logoHeight * scale) + 60; // Extra space for text
    
    // Get screen dimensions for centering
    SDL_DisplayMode dm;
    SDL_GetCurrentDisplayMode(0, &dm);
    int posX = (dm.w - m_windowWidth) / 2;
    int posY = (dm.h - m_windowHeight) / 2;
    
    // Create frameless window
    m_window = SDL_CreateWindow(
        "RayTrophi",
        posX, posY,
        m_windowWidth, m_windowHeight,
        SDL_WINDOW_BORDERLESS | SDL_WINDOW_SHOWN | SDL_WINDOW_SKIP_TASKBAR
    );
    
    if (m_window) {
        // Start transparent
        SDL_SetWindowOpacity(m_window, 0.0f);
    }
    
    if (!m_window) {
        std::cerr << "[SplashScreen] Failed to create window: " << SDL_GetError() << std::endl;
        SDL_FreeSurface(logoSurface);
        return false;
    }
    
    // Create renderer
    m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!m_renderer) {
        std::cerr << "[SplashScreen] Failed to create renderer: " << SDL_GetError() << std::endl;
        SDL_FreeSurface(logoSurface);
        return false;
    }
    
    // Create texture from logo
    m_logoTexture = SDL_CreateTextureFromSurface(m_renderer, logoSurface);
    SDL_FreeSurface(logoSurface);
    
    if (!m_logoTexture) {
        std::cerr << "[SplashScreen] Failed to create texture: " << SDL_GetError() << std::endl;
        return false;
    }
    
    m_initialized = true;
    render();
    render();
    
    // Fade in effect
    fadeIn(800);
    
    return true;
}

void SplashScreen::setStatus(const std::string& text) {
    m_statusText = text;
    render();
}

void SplashScreen::beginBusyStatus(const std::string& text) {
    m_busy = true;
    m_ready = false;
    m_busyStart = std::chrono::steady_clock::now();
    m_statusText = text;
    render();
}

void SplashScreen::stopBusyStatus() {
    m_busy = false;
    render();
}

void SplashScreen::tick() {
    if (!m_initialized) return;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            m_ready = false;
        }
    }

    render();
}

void SplashScreen::render() {
    if (!m_initialized || !m_renderer) return;
    
    // Clear with dark background
    SDL_SetRenderDrawColor(m_renderer, 20, 20, 25, 255);
    SDL_RenderClear(m_renderer);
    
    // Draw logo centered
    if (m_logoTexture) {
        int logoDisplayW = m_windowWidth;
        int logoDisplayH = m_windowHeight - 60;
        
        SDL_Rect destRect = {
            (m_windowWidth - logoDisplayW) / 2,
            0,
            logoDisplayW,
            logoDisplayH
        };
        SDL_RenderCopy(m_renderer, m_logoTexture, nullptr, &destRect);
    }
    
    // Draw status text background (bottom bar)
    SDL_SetRenderDrawColor(m_renderer, 30, 30, 35, 255);
    SDL_Rect bottomBar = { 0, m_windowHeight - 50, m_windowWidth, 50 };
    SDL_RenderFillRect(m_renderer, &bottomBar);
    
    // Draw border
    SDL_SetRenderDrawColor(m_renderer, 80, 80, 90, 255);
    SDL_Rect border = { 0, 0, m_windowWidth, m_windowHeight };
    SDL_RenderDrawRect(m_renderer, &border);
    
    const auto now = std::chrono::steady_clock::now();
    const auto busyMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_busyStart).count();
    const int elapsedSeconds = static_cast<int>(busyMs / 1000);

    // Draw animated status indicator
    SDL_Rect statusIndicator = { 20, m_windowHeight - 32, 10, 10 };
    if (m_ready) {
        SDL_SetRenderDrawColor(m_renderer, 0, 255, 150, 255);
        statusIndicator.w = 12;
        statusIndicator.h = 12;
    } else if (m_busy) {
        const Uint8 pulse = static_cast<Uint8>(140 + ((busyMs / 6) % 80));
        SDL_SetRenderDrawColor(m_renderer, 80, pulse, 120, 255);
    } else {
        SDL_SetRenderDrawColor(m_renderer, 100, 255, 100, 255);
    }
    SDL_RenderFillRect(m_renderer, &statusIndicator);

    // Draw activity bar so long operations look alive even without deterministic progress.
    SDL_Rect progressTrack = { 45, m_windowHeight - 20, m_windowWidth - 90, 4 };
    SDL_SetRenderDrawColor(m_renderer, 55, 55, 62, 255);
    SDL_RenderFillRect(m_renderer, &progressTrack);
    if (m_ready) {
        SDL_SetRenderDrawColor(m_renderer, 0, 255, 150, 255);
        SDL_RenderFillRect(m_renderer, &progressTrack);
    } else if (m_busy) {
        const int segmentWidth = (progressTrack.w / 5) > 40 ? (progressTrack.w / 5) : 40;
        const int travel = (progressTrack.w - segmentWidth) > 1 ? (progressTrack.w - segmentWidth) : 1;
        const int offset = static_cast<int>((busyMs / 10) % travel);
        SDL_SetRenderDrawColor(m_renderer, 90, 220, 160, 255);
        SDL_Rect movingSegment = { progressTrack.x + offset, progressTrack.y, segmentWidth, progressTrack.h };
        SDL_RenderFillRect(m_renderer, &movingSegment);
    }

    std::string statusLine = m_statusText;
    if (m_busy) {
        static const char* kFrames[] = {"-", "/", "-", "\\"};
        statusLine += " ";
        statusLine += kFrames[(busyMs / 200) % 4];
    }

    // Render status text string
    SDL_Color textColor = { 200, 200, 210, 255 };
    drawText(statusLine, 45, m_windowHeight - 35, textColor);
    if (m_busy) {
        const std::string timerText =
            std::to_string(elapsedSeconds / 60) + ":" +
            ((elapsedSeconds % 60) < 10 ? "0" : "") +
            std::to_string(elapsedSeconds % 60);

        SDL_SetRenderDrawColor(m_renderer, 24, 30, 34, 220);
        SDL_Rect timerBadge = { m_windowWidth - 96, 14, 78, 22 };
        SDL_RenderFillRect(m_renderer, &timerBadge);
        SDL_SetRenderDrawColor(m_renderer, 90, 220, 160, 255);
        SDL_RenderDrawRect(m_renderer, &timerBadge);

        SDL_Color accent = { 120, 220, 170, 255 };
        drawText(timerText, m_windowWidth - 87, 18, accent, 2);
    }
    
    SDL_RenderPresent(m_renderer);
}

void SplashScreen::waitForClick() {
    if (!m_initialized) return;
    
    m_ready = true;
    m_statusText = "Ready! Click to continue...";
    render();
    
    SDL_Event event;
    bool waiting = true;
    auto startTime = std::chrono::steady_clock::now();
    const auto maxWait = std::chrono::milliseconds(1000); // 1 second max
    
    while (waiting) {
        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - startTime;
        if (elapsed >= maxWait) {
            waiting = false;
            break;
        }
        
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                waiting = false;
            }
            if (event.type == SDL_KEYDOWN) {
                waiting = false; // Any key also works
            }
            if (event.type == SDL_QUIT) {
                waiting = false;
            }
        }
        SDL_Delay(16); // ~60fps
    }
    
    
    // Fade out before finishing
    fadeOut(500);
}


void SplashScreen::close() {
    if (m_logoTexture) {
        SDL_DestroyTexture(m_logoTexture);
        m_logoTexture = nullptr;
    }
    if (m_renderer) {
        SDL_DestroyRenderer(m_renderer);
        m_renderer = nullptr;
    }
    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    m_initialized = false;
}

void SplashScreen::drawText(const std::string& text, int x, int y, SDL_Color color, int scale) {
    if (!m_renderer || text.empty()) return;
    if (scale < 1) scale = 1;

    auto getCharData = [](char c) -> const unsigned char* {
        static const unsigned char fontData[][6] = {
            {'A', 0x7E, 0x11, 0x11, 0x11, 0x7E}, {'B', 0x7F, 0x49, 0x49, 0x49, 0x36},
            {'C', 0x3E, 0x41, 0x41, 0x41, 0x22}, {'D', 0x7F, 0x41, 0x41, 0x22, 0x1C},
            {'E', 0x7F, 0x49, 0x49, 0x49, 0x41}, {'F', 0x7F, 0x09, 0x09, 0x09, 0x01},
            {'G', 0x3E, 0x41, 0x49, 0x49, 0x7A}, {'H', 0x7F, 0x08, 0x08, 0x08, 0x7F},
            {'I', 0x00, 0x41, 0x7F, 0x41, 0x00}, {'J', 0x20, 0x40, 0x41, 0x3F, 0x01},
            {'K', 0x7F, 0x08, 0x14, 0x22, 0x41}, {'L', 0x7F, 0x40, 0x40, 0x40, 0x40},
            {'M', 0x7F, 0x02, 0x0C, 0x02, 0x7F}, {'N', 0x7F, 0x04, 0x08, 0x10, 0x7F},
            {'O', 0x3E, 0x41, 0x41, 0x41, 0x3E}, {'P', 0x7F, 0x09, 0x09, 0x09, 0x06},
            {'Q', 0x3E, 0x41, 0x51, 0x21, 0x5E}, {'R', 0x7F, 0x09, 0x19, 0x29, 0x46},
            {'S', 0x46, 0x49, 0x49, 0x49, 0x31}, {'T', 0x01, 0x01, 0x7F, 0x01, 0x01},
            {'U', 0x3F, 0x40, 0x40, 0x40, 0x3F}, {'V', 0x1F, 0x20, 0x40, 0x20, 0x1F},
            {'W', 0x3F, 0x40, 0x38, 0x40, 0x3F}, {'X', 0x63, 0x14, 0x08, 0x14, 0x63},
            {'Y', 0x07, 0x08, 0x70, 0x08, 0x07}, {'Z', 0x61, 0x51, 0x49, 0x45, 0x43},
            {'0', 0x3E, 0x51, 0x49, 0x45, 0x3E}, {'1', 0x00, 0x42, 0x7F, 0x40, 0x00},
            {'2', 0x42, 0x61, 0x51, 0x49, 0x46}, {'3', 0x21, 0x41, 0x45, 0x4B, 0x31},
            {'4', 0x18, 0x14, 0x12, 0x7F, 0x10}, {'5', 0x27, 0x45, 0x45, 0x45, 0x39},
            {'6', 0x3C, 0x4A, 0x49, 0x49, 0x30}, {'7', 0x01, 0x71, 0x09, 0x05, 0x03},
            {'8', 0x36, 0x49, 0x49, 0x49, 0x36}, {'9', 0x06, 0x49, 0x49, 0x29, 0x1E},
            {'.', 0x00, 0x60, 0x60, 0x00, 0x00}, {'!', 0x00, 0x00, 0x5F, 0x00, 0x00},
            {'?', 0x02, 0x01, 0x51, 0x09, 0x06}, {'-', 0x08, 0x08, 0x08, 0x08, 0x08},
            {':', 0x00, 0x36, 0x36, 0x00, 0x00}, {' ', 0x00, 0x00, 0x00, 0x00, 0x00},
            {',', 0x00, 0x50, 0x30, 0x00, 0x00}, {'_', 0x40, 0x40, 0x40, 0x40, 0x40},
            {'/', 0x20, 0x10, 0x08, 0x04, 0x02}, {'\\', 0x02, 0x04, 0x08, 0x10, 0x20},
            {'(', 0x1C, 0x22, 0x41, 0x00, 0x00}, {')', 0x00, 0x41, 0x22, 0x1C, 0x00}
        };
        for (const auto& entry : fontData) {
            if (entry[0] == c) return &entry[1];
        }
        return nullptr;
    };

    SDL_SetRenderDrawColor(m_renderer, color.r, color.g, color.b, color.a);
    
    int curX = x;
    for (char c : text) {
        const unsigned char* data = getCharData(toupper(c));
        if (data) {
            for (int i = 0; i < 5; i++) {
                unsigned char line = data[i];
                for (int j = 0; j < 8; j++) {
                    if (line & (1 << j)) {
                        SDL_Rect pixelRect = {
                            curX + (i * scale),
                            y + (j * scale),
                            scale,
                            scale
                        };
                        SDL_RenderFillRect(m_renderer, &pixelRect);
                    }
                }
            }
        }
        curX += 8 * scale;
    }
}

void SplashScreen::setOpacity(float opacity) {
    if (m_window) {
        SDL_SetWindowOpacity(m_window, opacity);
    }
}

void SplashScreen::fadeIn(int durationMs) {
    if (!m_window) return;
    
    auto startTime = std::chrono::steady_clock::now();
    bool fading = true;
    
    while (fading) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float, std::milli>(now - startTime).count();
        float progress = elapsed / durationMs;
        
        if (progress >= 1.0f) {
            progress = 1.0f;
            fading = false;
        }
        
        setOpacity(progress);
        
        // Keep checking events to keep window responsive (and allow early quit)
        SDL_Event e;
        while (SDL_PollEvent(&e)) {} 
        
        SDL_Delay(10);
    }
}

void SplashScreen::fadeOut(int durationMs) {
    if (!m_window) return;
    
    auto startTime = std::chrono::steady_clock::now();
    bool fading = true;
    
    while (fading) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float, std::milli>(now - startTime).count();
        float progress = 1.0f - (elapsed / durationMs);
        
        if (progress <= 0.0f) {
            progress = 0.0f;
            fading = false;
        }
        
        setOpacity(progress);
        
        // Keep checking events
        SDL_Event e;
        while (SDL_PollEvent(&e)) {}
        
        SDL_Delay(10);
    }
}
