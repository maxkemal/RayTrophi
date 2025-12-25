#pragma once

#include <SDL.h>
#include <string>

/**
 * SplashScreen - Frameless startup splash with logo and loading status
 * 
 * Usage:
 *   SplashScreen splash;
 *   splash.init("RayTrophi_image.png");
 *   splash.setStatus("Loading...");
 *   splash.render();
 *   splash.waitForClick();
 *   splash.close();
 */
class SplashScreen {
public:
    SplashScreen() = default;
    ~SplashScreen();

    // Initialize splash window with logo image
    bool init(const std::string& logoPath, int maxWidth = 800, int maxHeight = 600);
    
    // Update loading status text (bottom-left corner)
    void setStatus(const std::string& text);
    
    // Render current state (call after setStatus)
    void render();
    
    // Wait for mouse click to dismiss (blocking)
    void waitForClick();
    
    // Check if loading is marked as complete
    bool isReady() const { return m_ready; }
    
    // Mark loading as complete
    void setReady() { m_ready = true; }
    
    // Close and cleanup splash window
    void close();

private:
    SDL_Window* m_window = nullptr;
    SDL_Renderer* m_renderer = nullptr;
    SDL_Texture* m_logoTexture = nullptr;
    
    std::string m_statusText = "Loading...";
    std::string m_versionText = "v1.0.0";
    
    int m_windowWidth = 800;
    int m_windowHeight = 600;
    int m_logoWidth = 0;
    int m_logoHeight = 0;
    
    bool m_ready = false;
    bool m_initialized = false;
    
    // Simple bitmap font rendering (no SDL_ttf dependency)
    void drawText(const std::string& text, int x, int y, SDL_Color color);
};
