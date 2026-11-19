#include "Core/Localization.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <regex>
#include <stdexcept>

namespace Localization {
namespace {
using json = nlohmann::json;
struct State {
    bool loaded = false;
    std::string selected = "en";
    json manifest = {{"locales", json::array({{{"code", "en"}, {"name", "English"}}})}};
    std::map<std::string, json> catalogs;
    std::map<std::string, std::string> cache;
    std::map<std::string, std::string> aliases;
    std::string error;
};
State &state() {
    static State value;
    return value;
}
json read(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return json();
    }
    auto value = json::parse(stream, nullptr, false);
    return value.is_discarded() ? json() : value;
}
std::vector<std::string> printfArguments(const std::string &value) {
    std::vector<std::string> result;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '%' || i + 1 >= value.size()) {
            continue;
        }
        size_t next = i + 1;
        if (value[next] == '%') {
            ++i;
            continue;
        }
        while (next < value.size() &&
               std::string("-+ #0.123456789*").find(value[next]) != std::string::npos) {
            if (value[next] == '*') {
                result.emplace_back("*");
            }
            ++next;
        }
        const auto length = next;
        while (next < value.size() &&
               std::string("hlLjzt").find(value[next]) != std::string::npos) {
            ++next;
        }
        if (next < value.size() &&
            std::string("diuoxXfFeEgGaAcspn").find(value[next]) != std::string::npos) {
            result.push_back(value.substr(length, next - length + 1));
            i = next;
        }
    }
    return result;
}
std::set<std::string> namedArguments(const std::string &value) {
    static const std::regex placeholder(R"(\{([a-zA-Z_][a-zA-Z0-9_]*)\})");
    std::set<std::string> result;
    for (auto it = std::sregex_iterator(value.begin(), value.end(), placeholder);
         it != std::sregex_iterator(); ++it) {
        result.insert((*it)[1].str());
    }
    return result;
}
void load() {
    auto &s = state();
    if (s.loaded) {
        return;
    }
    s.loaded = true;
    for (const auto &root : {std::filesystem::path("assets/i18n"),
                             std::filesystem::path("../../RayTrophiStudio/assets/i18n")}) {
        const auto manifest = read(root / "locales.json");
        if (!manifest.is_object() || !manifest.contains("locales") ||
            !manifest["locales"].is_array()) {
            continue;
        }
        s.catalogs.clear();
        for (const auto &entry : manifest["locales"]) {
            if (!entry.is_object() || !entry.contains("code") || !entry["code"].is_string() ||
                !entry.contains("name") || !entry["name"].is_string()) {
                continue;
            }
            const auto code = entry["code"].get<std::string>();
            if (code.empty() ||
                code.find_first_not_of("abcdefghijklmnopqrstuvwxyz-") != std::string::npos) {
                continue;
            }
            const auto catalog = read(root / (code + ".json"));
            if (!catalog.is_object()) {
                continue;
            }
            bool valid = true;
            for (const auto &item : catalog.items()) {
                valid &= item.value().is_string();
            }
            if (valid) {
                s.catalogs.emplace(code, catalog);
            }
        }
        if (s.catalogs.contains("en")) {
            for (auto &catalog : s.catalogs) {
                if (catalog.first == "en") {
                    continue;
                }
                for (auto it = catalog.second.begin(); it != catalog.second.end();) {
                    if (s.catalogs["en"].contains(it.key())) {
                        const auto english = s.catalogs["en"][it.key()].get<std::string>();
                        const auto translated = it.value().get<std::string>();
                        if (printfArguments(english) != printfArguments(translated) ||
                            namedArguments(english) != namedArguments(translated)) {
                            s.error = "Unsafe translation placeholders rejected; English fallback "
                                      "is active";
                            it = catalog.second.erase(it);
                            continue;
                        }
                    }
                    ++it;
                }
            }
            s.manifest = manifest;
            for (const auto &item : s.catalogs["en"].items()) {
                s.aliases.emplace(item.value().get<std::string>(), item.key());
            }
            return;
        }
    }
    s.catalogs.clear();
    s.error = "Translation catalogs are missing; built-in English fallback is active";
}
const std::string &resolve(const std::string &key, const char *english, bool id) {
    load();
    auto &s = state();
    const auto token = s.selected + (id ? "|widget|" : "|text|") + key;
    auto found = s.cache.find(token);
    if (found != s.cache.end()) {
        return found->second;
    }
    std::string value = english;
    for (const auto &code : {s.selected, std::string("en")}) {
        const auto catalog = s.catalogs.find(code);
        if (catalog != s.catalogs.end() && catalog->second.contains(key)) {
            value = catalog->second[key].get<std::string>();
            break;
        }
    }
    if (id) {
        value += "###" + key;
    }
    return s.cache.emplace(token, std::move(value)).first->second;
}
} // namespace
const char *text(const std::string &key, const char *english) {
    return resolve(key, english, false).c_str();
}
const char *widget(const std::string &key, const char *english) {
    return resolve(key, english, true).c_str();
}
std::string message(const std::string &english) {
    load();
    const auto &aliases = state().aliases;
    const auto found = aliases.find(english);
    if (found != aliases.end()) {
        return text(found->second, english.c_str());
    }
    // Legacy diagnostic messages append paths/field names. Only translate a
    // registered prefix, never the caller's project/device names or raw values.
    const std::pair<const std::string, std::string> *prefix = nullptr;
    for (const auto &item : aliases) {
        if (!item.first.empty() && item.first.size() < english.size() &&
            english.starts_with(item.first) &&
            (item.first.back() == ' ' || item.first.back() == '.') &&
            (!prefix || prefix->first.size() < item.first.size())) {
            prefix = &item;
        }
    }
    return prefix ? std::string(text(prefix->second, prefix->first.c_str())) +
                        message(english.substr(prefix->first.size()))
                  : english;
}
std::string format(const std::string &key, const char *english, const json &arguments) {
    if (!arguments.is_object()) {
        throw std::invalid_argument("Localization arguments must be an object");
    }
    const std::string pattern = text(key, english);
    std::string result;
    for (size_t i = 0; i < pattern.size();) {
        if (pattern[i] != '{') {
            result += pattern[i++];
            continue;
        }
        const auto end = pattern.find('}', i + 1);
        if (end == std::string::npos) {
            throw std::invalid_argument("Unclosed localization placeholder");
        }
        const auto name = pattern.substr(i + 1, end - i - 1);
        if (!arguments.contains(name)) {
            throw std::invalid_argument("Missing localization argument: " + name);
        }
        result += arguments[name].is_string() ? arguments[name].get<std::string>()
                                              : arguments[name].dump();
        i = end + 1;
    }
    return result;
}
bool setLocale(const std::string &code, std::string &error) {
    error.clear();
    load();
    if (code != "en" && !state().catalogs.contains(code)) {
        error = "Unsupported or unavailable locale: " + code;
        return false;
    }
    state().selected = code;
    return true;
}
const std::string &locale() {
    load();
    return state().selected;
}
json status() {
    load();
    auto available = json::array();
    for (const auto &item : state().manifest["locales"]) {
        if (item.is_object() && item.contains("code") && item["code"].is_string() &&
            item.contains("name") && item["name"].is_string() &&
            (item["code"] == "en" || state().catalogs.contains(item["code"].get<std::string>()))) {
            available.push_back(item);
        }
    }
    return {{"locale", locale()}, {"available", available}, {"error", state().error}};
}
} // namespace Localization
