#pragma once
#include <string>
struct SDL_Renderer;
namespace AppRuntime {
void showPlayerOperator(const std::string &page = "games");
bool playerOperatorOpen();
void closePlayerOperator();
void drawPlayerOperator(SDL_Renderer *renderer);
void loadPlayerOperatorFonts();
void releasePlayerOperatorUI();
} // namespace AppRuntime
