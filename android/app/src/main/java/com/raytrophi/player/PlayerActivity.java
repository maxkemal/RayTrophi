package com.raytrophi.player;

import android.os.Bundle;
import android.view.WindowManager;

import org.libsdl.app.SDLActivity;

/**
 * Entry activity of the RayTrophi Player. SDL owns the surface, the Vulkan window and input;
 * the native game loop is PlayerMain.cpp compiled into libmain.so (SDL_main).
 * The org.libsdl.app.* classes come from the SDL2 source tree (see android/README.md).
 */
public class PlayerActivity extends SDLActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // SDL2 hides the system bars for a fullscreen window but leaves the display cutout mode
        // at its default, which letterboxes a landscape window away from a camera hole: a black
        // strip on that short edge (observed on an Omix X700). Draw edge to edge instead.
        // minSdk is 29, so the API 28 attribute is always available.
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        attributes.layoutInDisplayCutoutMode =
            WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        getWindow().setAttributes(attributes);
        super.onCreate(savedInstanceState);
    }

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
