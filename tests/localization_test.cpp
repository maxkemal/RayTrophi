#include "../RayTrophiStudio/source/include/Core/Localization.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>

// Standalone user-run test: link Core/Localization.cpp; run from repository root.
// Codex does not compile or execute C++ tests in this workspace.
int main() {
    std::filesystem::current_path("RayTrophiStudio");
    std::string error;
    assert(Localization::setLocale("en", error));
    const auto *stable = Localization::text("player.save", "Save");
    const std::string english_id = Localization::widget("player.save", "Save");
    assert(std::string(stable) == "Save");
    assert(Localization::setLocale("tr", error));
    std::ifstream stream("assets/i18n/tr.json");
    const auto translated = nlohmann::json::parse(stream);
    assert(Localization::text("player.save", "Save") == translated["player.save"]);
    const std::string turkish_id = Localization::widget("player.save", "Save");
    assert(english_id.substr(english_id.find("###")) == turkish_id.substr(turkish_id.find("###")));
    assert(std::string(stable) == "Save"); // Switching locale never invalidates cached pointers.
    assert(std::string(Localization::text("missing.test", "English fallback")) ==
           "English fallback");
    assert(!Localization::setLocale("../invalid", error));
    assert(Localization::locale() == "tr");
    const auto value = Localization::format("test.named", "Hello {name} / {count}",
                                            {{"name", "{count}"}, {"count", 3}});
    assert(value == "Hello {count} / 3"); // Argument text is never expanded as a template.
    bool failed = false;
    try {
        Localization::format("test.missing", "{required}", nlohmann::json::object());
    } catch (const std::invalid_argument &) {
        failed = true;
    }
    assert(failed);
}
