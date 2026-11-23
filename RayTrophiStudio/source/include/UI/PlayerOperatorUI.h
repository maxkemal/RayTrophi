#pragma once
#include <string>
struct SDL_Renderer;
namespace AppRuntime {
void showPlayerOperator(const std::string &page = "games");
bool playerOperatorOpen();
std::string playerOperatorPage();
void closePlayerOperator();
void drawPlayerOperator(SDL_Renderer *renderer);
void loadPlayerOperatorFonts();
void releasePlayerOperatorUI();
// Touch layout scale (1 on Windows). On Android playerOperatorOpen() is also true while the kid
// launcher owns the screen; playerConsoleOpen() is the operator console alone.
float playerOperatorScale();
bool playerConsoleOpen();
} // namespace AppRuntime
