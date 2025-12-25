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
        SDL_WINDOW_BORDERLESS | SDL_WINDOW_SHOWN | SDL_WINDOW_ALWAYS_ON_TOP
    );
    
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
    return true;
}

void SplashScreen::setStatus(const std::string& text) {
    m_statusText = text;
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
    
    // Draw text using simple colored rectangles (no SDL_ttf needed)
    // Status text indicator - left side
    SDL_SetRenderDrawColor(m_renderer, 100, 200, 100, 255);
    SDL_Rect statusIndicator = { 10, m_windowHeight - 35, 8, 8 };
    SDL_RenderFillRect(m_renderer, &statusIndicator);
    
    // "Ready" indicator changes color
    if (m_ready) {
        SDL_SetRenderDrawColor(m_renderer, 50, 255, 100, 255);
        SDL_Rect readyIndicator = { 10, m_windowHeight - 35, 12, 12 };
        SDL_RenderFillRect(m_renderer, &readyIndicator);
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

void SplashScreen::drawText(const std::string& text, int x, int y, SDL_Color color) {
    // Placeholder - SDL_ttf would be needed for actual text
    // For now, we just show colored indicators
}
