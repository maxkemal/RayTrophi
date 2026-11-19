#pragma once
#include "json.hpp"
#include <string>

// UI-independent, main-thread localization shared by executable hosts.
namespace Localization {
const char *text(const std::string &key, const char *english);
const char *widget(const std::string &key, const char *english);
std::string message(const std::string &english);
std::string format(const std::string &key, const char *english, const nlohmann::json &arguments);
bool setLocale(const std::string &locale, std::string &error);
const std::string &locale();
nlohmann::json status();
} // namespace Localization
