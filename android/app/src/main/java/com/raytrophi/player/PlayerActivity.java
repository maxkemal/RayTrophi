package com.raytrophi.player;

import org.libsdl.app.SDLActivity;

/**
 * Entry activity of the RayTrophi Player. SDL owns the surface, the Vulkan window and input;
 * the native game loop is PlayerMain.cpp compiled into libmain.so (SDL_main).
 * The org.libsdl.app.* classes come from the SDL2 source tree (see android/README.md).
 */
public class PlayerActivity extends SDLActivity {
    @Override
    protected String[] getLibraries() {
        // Order matters: dependencies first, the SDL_main library ("main") last.
        return new String[] {
            "c++_shared",
            "SDL2",
            "SDL2_image",
            "main"
        };
    }
}
