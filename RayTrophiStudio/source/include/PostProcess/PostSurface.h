#pragma once
#include "SDL.h"
class ColorProcessor;
class Renderer;
class Camera;
class Vec3;
Vec3 applyVignette(const Vec3&,int,int,int,int,float strength=1.0f);
void applyToneMappingToSurfaceWithCamera(SDL_Surface*,SDL_Surface*,ColorProcessor&,Renderer*,const Camera*);
void applyToneMappingToSurface(SDL_Surface*,SDL_Surface*,ColorProcessor&,Renderer*);
void applyStylizeToSurfaceWithCamera(SDL_Surface*,Renderer&,bool,const Camera*);
void applyStylizeToSurface(SDL_Surface*,Renderer&,bool);
void applyCPUDenoisedPreviewToSurface(SDL_Surface*,Renderer&,const Camera*);
