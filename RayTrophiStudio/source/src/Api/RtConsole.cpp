/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtConsole.cpp
 * Date:          July 2026
 * License:       MIT
 * =========================================================================
 */
#include "Api/RtPython.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "TextEditor.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

#include "imgui.h"

// Reference global UI context pointer defined in RtApi.cpp
namespace rtapi {
    extern UIContext* g_ctx;
}

namespace rtpython {
namespace {

// Text Editor State
TextEditor g_editor;
bool g_editor_initialized = false;

// Autocomplete State
bool g_autocomplete_active = false;
std::vector<std::string> g_autocomplete_items;
int g_autocomplete_index = 0;
std::string g_autocomplete_prefix = "";
std::string g_autocomplete_mod_path = "";

// Buffers
std::array<char, 8192> g_input{};
std::vector<std::string> g_history;
int g_history_position = -1;
bool g_scroll_to_bottom = true;
bool g_wants_input_capture = false;

// File Management State
std::string g_current_script_path = "";
bool g_editor_dirty = false;

// Splitter State
float g_editor_width_ratio = 0.72f; // Default editor width ratio
float g_editor_height_ratio = 0.55f; // Default editor height ratio
float g_editor_zoom = 1.0f;          // Editor font zoom scale

// Native Win32 File Dialog helpers
std::string wstring_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

// Native Win32 File Dialog helpers (Wide/Unicode character versions)
std::string openFileDialog() {
#ifdef _WIN32
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn = { 0 };
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = L"Python Files\0*.py\0All Files\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle = L"Open Python Script";
    ofn.hwndOwner = GetActiveWindow();

    if (GetOpenFileNameW(&ofn)) {
        return wstring_to_utf8(filename);
    }
#endif
    return "";
}

std::string saveFileDialog() {
#ifdef _WIN32
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn = { 0 };
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = L"Python Files\0*.py\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle = L"Save Python Script";
    ofn.hwndOwner = GetActiveWindow();

    if (GetSaveFileNameW(&ofn)) {
        std::wstring path = filename;
        if (path.size() < 3 || path.substr(path.size() - 3) != L".py") {
            path += L".py";
        }
        return wstring_to_utf8(path);
    }
#endif
    return "";
}

// File I/O Logic using std::filesystem::path (handles UTF-8 paths natively on Windows)
void loadScript(const std::string& path) {
    if (path.empty()) return;
    std::ifstream file(std::filesystem::path(path), std::ios::in | std::ios::binary);
    if (file.is_open()) {
        std::stringstream ss;
        ss << file.rdbuf();
        std::string content = ss.str();

        g_editor.SetText(content);
        g_current_script_path = path;
        g_editor_dirty = false;
        appendConsoleText("[Workspace] Loaded: " + path + "\n");
    } else {
        appendConsoleText("[Workspace] Error: Failed to open file " + path + "\n");
    }
}

void saveScript(const std::string& path) {
    if (path.empty()) return;
    std::ofstream file(std::filesystem::path(path), std::ios::out | std::ios::binary);
    if (file.is_open()) {
        const std::string content = g_editor.GetText();
        file.write(content.c_str(), content.size());
        g_current_script_path = path;
        g_editor_dirty = false;
        appendConsoleText("[Workspace] Saved: " + path + "\n");
    } else {
        appendConsoleText("[Workspace] Error: Failed to save file " + path + "\n");
    }
}

int historyCallback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory || g_history.empty()) return 0;
    const int previous = g_history_position;
    if (data->EventKey == ImGuiKey_UpArrow) {
        if (g_history_position < 0) g_history_position = static_cast<int>(g_history.size()) - 1;
        else if (g_history_position > 0) --g_history_position;
    } else if (data->EventKey == ImGuiKey_DownArrow) {
        if (g_history_position >= 0 && ++g_history_position >= static_cast<int>(g_history.size())) {
            g_history_position = -1;
        }
    }
    if (previous != g_history_position) {
        const std::string text = g_history_position >= 0 ? g_history[g_history_position] : std::string{};
        data->DeleteChars(0, data->BufTextLen);
        data->InsertChars(0, text.c_str());
    }
    return 0;
}

// Native IntelliSense / Tab Autocomplete Callback
int replInputCallback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
        return historyCallback(data);
    }

    if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
        std::string buf = data->Buf;
        int cursor = data->CursorPos;
        if (cursor <= 0) return 0;

        // Find start boundary of completion word
        int start = cursor - 1;
        while (start >= 0 && (std::isalnum(buf[start]) || buf[start] == '.' || buf[start] == '_')) {
            --start;
        }
        ++start;

        std::string word = buf.substr(start, cursor - start);
        if (word.empty()) return 0;

        // E.g., "rt.scene.o" -> module "rt.scene", prefix "o"
        size_t last_dot = word.find_last_of('.');
        std::string mod_path = "rt";
        std::string prefix = word;

        if (last_dot != std::string::npos) {
            mod_path = word.substr(0, last_dot);
            prefix = word.substr(last_dot + 1);
        } else {
            mod_path = "";
            prefix = word;
        }

        std::vector<std::string> attrs = getModuleAttributes(mod_path);
        std::vector<std::string> matches;
        for (const auto& attr : attrs) {
            if (attr.rfind(prefix, 0) == 0) {
                matches.push_back(attr);
            }
        }

        if (matches.empty()) return 0;

        if (matches.size() == 1) {
            std::string completion = matches[0];
            data->DeleteChars(cursor - prefix.size(), prefix.size());
            data->InsertChars(data->CursorPos, completion.c_str());

            // Check if submodule -> add dot, else add bracket for functions.
            // Reflected from the live rt module so new submodules complete correctly.
            bool isSubmodule = false;
            for (const auto& path : getSubmodulePaths()) {
                if (path.rfind("rt.", 0) == 0 && path.compare(3, std::string::npos, completion) == 0) {
                    isSubmodule = true;
                    break;
                }
            }
            if (isSubmodule) {
                data->InsertChars(data->CursorPos, ".");
            } else {
                data->InsertChars(data->CursorPos, "()");
                // Back character to place cursor inside brackets
                data->CursorPos--;
            }
        } else {
            // Output completion hints to console logs
            appendConsoleText("\nSuggestions:\n");
            int col_count = 0;
            for (const auto& match : matches) {
                appendConsoleText("  " + match);
                if (++col_count % 3 == 0) appendConsoleText("\n");
            }
            if (col_count % 3 != 0) appendConsoleText("\n");
            appendConsoleText(">>> " + std::string(data->Buf) + "\n");
            g_scroll_to_bottom = true;
        }
    }
    return 0;
}

void runInput() {
    std::string source = g_input.data();
    if (source.empty()) return;
    g_history.erase(std::remove(g_history.begin(), g_history.end(), source), g_history.end());
    g_history.push_back(source);
    g_history_position = -1;

    // Echo commands independently of redirected stdout.
    appendConsoleText(">>> " + source + "\n");
    execute(source);
    g_input.fill('\0');
    g_scroll_to_bottom = true;
}

void runEditorCode() {
    const std::string code = g_editor.GetText();
    if (code.empty()) {
        appendConsoleText("[Workspace] Editor is empty.\n");
        return;
    }
    appendConsoleText("[Workspace] Running script...\n");

    ExecutionResult result = execute(code, g_current_script_path.empty() ? "<editor>" : g_current_script_path);
    if (result.ok) {
        appendConsoleText("[Workspace] Script execution completed successfully.\n");
    } else {
        appendConsoleText("[Workspace] Script failed with error.\n");
    }
}

void insertTextIntoEditor(const std::string& text) {
    g_editor.InsertText(text);
    g_editor_dirty = true;
}

void triggerAutocomplete() {
    if (!isInitialized()) return;

    TextEditor::Coordinates cursor = g_editor.GetCursorPosition();
    std::string line = g_editor.GetCurrentLineText();

    // Find start boundary of completion word
    int col = cursor.mColumn;
    if (col <= 0 || col > (int)line.size()) return;

    int start = col - 1;
    while (start >= 0 && (std::isalnum(line[start]) || line[start] == '.' || line[start] == '_')) {
        --start;
    }
    ++start;

    std::string word = line.substr(start, col - start);
    if (word.empty()) return;

    size_t last_dot = word.find_last_of('.');
    std::string mod_path = "";
    std::string prefix = word;

    if (last_dot != std::string::npos) {
        mod_path = word.substr(0, last_dot);
        prefix = word.substr(last_dot + 1);
    } else {
        mod_path = "";
        prefix = word;
    }

    std::vector<std::string> attrs = getModuleAttributes(mod_path);
    std::vector<std::string> matches;
    for (const auto& attr : attrs) {
        if (attr.rfind(prefix, 0) == 0) {
            matches.push_back(attr);
        }
    }

    if (!matches.empty()) {
        g_autocomplete_active = true;
        g_autocomplete_items = std::move(matches);
        g_autocomplete_index = 0;
        g_autocomplete_prefix = prefix;
        g_autocomplete_mod_path = mod_path;
    } else {
        g_autocomplete_active = false;
    }
}

// Colorized Output Printer (Syntax Coloring for logs)
void drawColorizedLogs(const std::string& text) {
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.rfind(">>>", 0) == 0) {
            // Command input lines
            ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "%s", line.c_str());
        } else if (line.find("Error:") != std::string::npos || line.find("Traceback") != std::string::npos || line.find("FAIL") != std::string::npos) {
            // Failure and traceback logs
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", line.c_str());
        } else if (line.find("[rt-smoke] PASS") != std::string::npos || line.find("success") != std::string::npos || line.find("OK") != std::string::npos) {
            // Pass and success confirmations
            ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.5f, 1.0f), "%s", line.c_str());
        } else if (line.rfind("#", 0) == 0 || line.rfind("[Workspace]", 0) == 0) {
            // Workspace log system messages or comments
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", line.c_str());
        } else {
            // Standard standard stdout
            ImGui::TextUnformatted(line.c_str());
        }
    }
}

} // namespace

void drawConsole(bool* open) {
    if (!open || !*open) {
        g_wants_input_capture = false;
        return;
    }

    if (!g_editor_initialized) {
        g_editor.SetLanguageDefinition(TextEditor::LanguageDefinition::Python());
        g_editor.SetPalette(TextEditor::GetDarkPalette());
        g_editor_initialized = true;
    }

    ImGui::SetNextWindowSize(ImVec2(1100.0f, 650.0f), ImGuiCond_FirstUseEver);
    const bool visible = ImGui::Begin("Python Scripting Workspace", open, ImGuiWindowFlags_MenuBar);
    g_wants_input_capture =
        ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) ||
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

    if (!visible) {
        ImGui::End();
        return;
    }

    const bool ready = isInitialized();

    // ── MENU BAR ───────────────────────────────────────────────────────────
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Script", "Ctrl+N")) {
                g_editor.SetText("");
                g_current_script_path = "";
                g_editor_dirty = false;
            }
            if (ImGui::MenuItem("Open Script...", "Ctrl+O")) {
                std::string path = openFileDialog();
                if (!path.empty()) loadScript(path);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save Script", "Ctrl+S")) {
                if (g_current_script_path.empty()) {
                    std::string path = saveFileDialog();
                    if (!path.empty()) saveScript(path);
                } else {
                    saveScript(g_current_script_path);
                }
            }
            if (ImGui::MenuItem("Save Script As...")) {
                std::string path = saveFileDialog();
                if (!path.empty()) saveScript(path);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Run")) {
            if (ImGui::MenuItem("Run Script", "F5", nullptr, ready)) {
                runEditorCode();
            }
            ImGui::EndMenu();
        }

        // Display current active file path
        ImGui::Separator();
        if (!g_current_script_path.empty()) {
            ImGui::TextDisabled("File: %s%s", g_current_script_path.c_str(), g_editor_dirty ? " *" : "");
        } else {
            ImGui::TextDisabled("File: Untitled.py%s", g_editor_dirty ? " *" : "");
        }

        // Right-aligned settings quick-access
        float width = ImGui::GetWindowWidth();
        if (width > 800.0f) {
            ImGui::SameLine(width - 260.0f);
            if (rtapi::g_ctx) {
                ImGui::SetNextItemWidth(80.0f);
                ImGui::SliderInt("SPP", &rtapi::g_ctx->render_settings.final_render_samples, 1, 1024);
                ImGui::SameLine();
                ImGui::TextDisabled("Mode: PT");
            }
        }

        ImGui::EndMenuBar();
    }

    // ── MAIN SPLIT RESIZING LAYOUT ─────────────────────────────────────────
    const float availWidth = ImGui::GetContentRegionAvail().x;
    const float editorWidth = availWidth * g_editor_width_ratio;
    const float explorerWidth = availWidth - editorWidth - 16.0f; // Splitter space offset

    // ── LEFT GROUP: EDITOR & CONSOLE ───────────────────────────────────────
    ImGui::BeginGroup();
    {
        ImGui::TextColored(ImVec4(0.35f, 0.7f, 1.0f, 1.0f), "Python Script Editor");
        ImGui::SameLine();
        if (ImGui::SmallButton("Run Code (F5)")) {
            runEditorCode();
        }

        // Multi-line editor buffer (responsive height adjustable via splitter)
        const float totalHeight = ImGui::GetContentRegionAvail().y;
        const float editorHeight = totalHeight * g_editor_height_ratio;

        // Zoom functionality via Ctrl + Mouse Wheel when editor is focused
        if (g_editor.IsFocused()) {
            if (ImGui::GetIO().KeyCtrl) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    g_editor_zoom += wheel * 0.05f;
                    g_editor_zoom = std::clamp(g_editor_zoom, 0.5f, 2.5f);
                }
            }
        }

        // Render the advanced text editor with local font zoom scale applied
        float old_scale = ImGui::GetFont()->Scale;
        ImGui::GetFont()->Scale = g_editor_zoom;

        g_editor.Render("##ScriptCode", ImVec2(editorWidth, editorHeight));

        ImGui::GetFont()->Scale = old_scale;

        if (g_editor.IsTextChanged()) {
            g_editor_dirty = true;
        }

        // Autocomplete logic
        if (g_editor.IsFocused()) {
            bool ctrl_space = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyPressed(ImGuiKey_Space);
            bool dot_pressed = false;

            if (g_editor.IsTextChanged()) {
                TextEditor::Coordinates cursor = g_editor.GetCursorPosition();
                std::string line = g_editor.GetCurrentLineText();
                int col = cursor.mColumn;
                if (col > 0 && col <= (int)line.size() && line[col - 1] == '.') {
                    dot_pressed = true;
                }
            }

            if (ctrl_space || dot_pressed) {
                triggerAutocomplete();
            }
        }

        // Render Autocomplete Popup
        if (g_autocomplete_active && !g_autocomplete_items.empty()) {
            ImVec2 popup_pos = g_editor.mCursorScreenPos;
            popup_pos.y += ImGui::GetTextLineHeightWithSpacing(); // Position below cursor

            ImGui::SetNextWindowPos(popup_pos);
            ImGui::SetNextWindowSizeConstraints(ImVec2(180, 0), ImVec2(400, 300));

            ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.65f, 1.0f, 1.0f));

            if (ImGui::Begin("##AutocompletePopup", nullptr, flags)) {
                // Keyboard navigations
                if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
                    g_autocomplete_index = (g_autocomplete_index + 1) % g_autocomplete_items.size();
                }
                if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
                    g_autocomplete_index = (g_autocomplete_index - 1 + g_autocomplete_items.size()) % g_autocomplete_items.size();
                }

                // Cancel
                if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                    g_autocomplete_active = false;
                }

                // Accept
                bool select_item = false;
                if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_Tab)) {
                    select_item = true;
                }

                // List matching items
                for (int i = 0; i < (int)g_autocomplete_items.size(); ++i) {
                    const bool is_selected = (i == g_autocomplete_index);
                    if (ImGui::Selectable(g_autocomplete_items[i].c_str(), is_selected)) {
                        g_autocomplete_index = i;
                        select_item = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }

                // Apply completion on selection
                if (select_item && g_autocomplete_index >= 0 && g_autocomplete_index < (int)g_autocomplete_items.size()) {
                    std::string completion = g_autocomplete_items[g_autocomplete_index];

                    TextEditor::Coordinates cursor = g_editor.GetCursorPosition();
                    TextEditor::Coordinates start = cursor;
                    start.mColumn = (std::max)(0, start.mColumn - (int)g_autocomplete_prefix.size());

                    g_editor.SetSelection(start, cursor);
                    g_editor.InsertText(completion);

                    // Check if function or module and add parenthesis or dot
                    bool isSubmodule = false;
                    for (const auto& path : getSubmodulePaths()) {
                        if (path.rfind("rt.", 0) == 0 && path.compare(3, std::string::npos, completion) == 0) {
                            isSubmodule = true;
                            break;
                        }
                    }
                    if (!isSubmodule && completion != "rt") {
                        g_editor.InsertText("()");
                        TextEditor::Coordinates c = g_editor.GetCursorPosition();
                        c.mColumn--;
                        g_editor.SetCursorPosition(c);
                    } else {
                        g_editor.InsertText(".");
                    }

                    g_autocomplete_active = false;
                }
            }
            ImGui::End();
            ImGui::PopStyleColor(2);

            // Render documentation balloon next to the popup
            if (g_autocomplete_active && g_autocomplete_index >= 0 && g_autocomplete_index < (int)g_autocomplete_items.size()) {
                std::string completion = g_autocomplete_items[g_autocomplete_index];
                std::string full_path = g_autocomplete_mod_path.empty() ? completion : (g_autocomplete_mod_path + "." + completion);
                std::string doc = getAttributeDocstring(g_autocomplete_mod_path, completion);
                if (!doc.empty()) {
                    ImGui::SetNextWindowPos(ImVec2(popup_pos.x + 185.0f, popup_pos.y));
                    ImGui::SetNextWindowSizeConstraints(ImVec2(200, 0), ImVec2(500, 300));
                    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.05f, 0.95f));
                    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.65f, 1.0f, 0.8f));
                    if (ImGui::Begin("##AutocompleteDoc", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings)) {
                        ImGui::TextColored(ImVec4(0.35f, 0.7f, 1.0f, 1.0f), "%s", full_path.c_str());
                        ImGui::Separator();
                        ImGui::TextWrapped("%s", doc.c_str());
                    }
                    ImGui::End();
                    ImGui::PopStyleColor(2);
                }
            }
        }

        // ── INTERACTIVE HORIZONTAL SPLITTER ────────────────────────────────────
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.7f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));

        ImGui::Button("##workspace_height_splitter", ImVec2(editorWidth, 6.0f));
        if (ImGui::IsItemActive()) {
            g_editor_height_ratio += ImGui::GetIO().MouseDelta.y / totalHeight;
            g_editor_height_ratio = std::clamp(g_editor_height_ratio, 0.20f, 0.80f);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        ImGui::PopStyleColor(3);

        ImGui::Spacing();

        // REPL & Log Header
        ImGui::TextColored(ImVec4(0.35f, 0.9f, 0.55f, 1.0f), "REPL Output & Console Log");
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear")) clearConsoleOutput();
        ImGui::SameLine();
        if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(consoleOutputSnapshot().c_str());

        const float bottomHeight = ImGui::GetContentRegionAvail().y - 45.0f;
        if (ImGui::BeginChild("##PythonOutputWorkspace", ImVec2(editorWidth, bottomHeight), true,
                              ImGuiWindowFlags_HorizontalScrollbar)) {
            const std::string output = consoleOutputSnapshot();
            // Capture bottom-pinned state BEFORE drawing new content: once content
            // is appended MaxY grows while ScrollY lags a frame, so measuring after
            // would miss the "was at bottom" case and never auto-scroll.
            const bool was_at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
            drawColorizedLogs(output);

            // Auto-scroll on new output only when the user is already at the bottom,
            // so scrolling up to read older logs is never hijacked. g_scroll_to_bottom
            // still forces a jump right after a REPL command.
            if (g_scroll_to_bottom || was_at_bottom) {
                ImGui::SetScrollHereY(1.0f);
            }
            g_scroll_to_bottom = false;
        }
        ImGui::EndChild();

        // Interactive REPL Input Line with native autocompletion flag
        const ImGuiInputTextFlags replFlags =
            ImGuiInputTextFlags_EnterReturnsTrue |
            ImGuiInputTextFlags_CallbackHistory |
            ImGuiInputTextFlags_CallbackCompletion;

        ImGui::SetNextItemWidth(editorWidth - 110.0f);
        const bool submitted = ImGui::InputText("##PythonReplInput", g_input.data(), g_input.size(), replFlags,
                                                replInputCallback);
        ImGui::SameLine();
        const bool runClicked = ImGui::Button("Execute", ImVec2(100.0f, 0.0f));
        if (ready && (submitted || runClicked)) {
            runInput();
            ImGui::SetKeyboardFocusHere(-1);
        }
    }
    ImGui::EndGroup();

    ImGui::SameLine();

    // ── INTERACTIVE VERTICAL SPLITTER ──────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.7f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));

    ImGui::Button("##workspace_splitter", ImVec2(6.0f, ImGui::GetContentRegionAvail().y));
    if (ImGui::IsItemActive()) {
        g_editor_width_ratio += ImGui::GetIO().MouseDelta.x / availWidth;
        g_editor_width_ratio = std::clamp(g_editor_width_ratio, 0.35f, 0.85f);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // ── RIGHT GROUP: API EXPLORER ──────────────────────────────────────────
    ImGui::BeginGroup();
    {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.35f, 1.0f), "rt API Explorer");

        ImGui::BeginChild("##ApiExplorerPanel", ImVec2(explorerWidth, 0.0f), true);
        {
            if (!ready) {
                ImGui::TextWrapped("Python needs to be initialized to reflect standard API.");
            } else {
                // Addons (Faz 4a): enable/disable/reload discovered addons. The
                // enabled state persists across launches (addon_state.json).
                if (ImGui::CollapsingHeader("Addons")) {
                    const std::vector<AddonInfo> addons = listAddons();
                    if (addons.empty()) {
                        ImGui::TextDisabled("None in scripts/addons/");
                    }
                    for (const auto& a : addons) {
                        ImGui::PushID(a.module_name.c_str());
                        bool enabled = a.enabled;
                        const std::string label = a.display_name.empty() ? a.module_name : a.display_name;
                        if (ImGui::Checkbox(label.c_str(), &enabled)) {
                            std::string err;
                            const bool ok = enabled ? enableAddon(a.module_name, err)
                                                    : disableAddon(a.module_name, err);
                            if (!ok) appendConsoleText("Addon error: " + err + "\n");
                        }
                        if (ImGui::IsItemHovered() && !a.description.empty()) {
                            ImGui::SetTooltip("%s", a.description.c_str());
                        }
                        if (a.loaded) {
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Reload")) {
                                std::string err;
                                if (!reloadAddon(a.module_name, err))
                                    appendConsoleText("Addon reload error: " + err + "\n");
                            }
                        }
                        ImGui::PopID();
                    }
                    ImGui::Separator();
                }

                // Reflected from the live rt module, so new submodules (rt.mesh,
                // rt.anim, rt.nodes, ...) appear here automatically.
                const std::vector<std::string> submodules = getSubmodulePaths();

                for (const auto& mod : submodules) {
                    if (ImGui::TreeNode(mod.c_str())) {
                        std::vector<std::string> attrs = getModuleAttributes(mod);
                        if (attrs.empty()) {
                            ImGui::TextDisabled("No attributes or functions");
                        } else {
                            for (const auto& attr : attrs) {
                                std::string type = getAttributeType(mod, attr);
                                std::string item_name;
                                ImVec4 color;
                                bool is_fn = false;

                                if (type == "class") {
                                    item_name = "[C] " + attr;
                                    color = ImVec4(1.0f, 0.6f, 0.3f, 1.0f); // Orange
                                } else if (type == "method") {
                                    item_name = "[M] " + attr + "()";
                                    color = ImVec4(0.35f, 0.7f, 1.0f, 1.0f); // Light blue
                                    is_fn = true;
                                } else if (type == "module") {
                                    item_name = "[S] " + attr;
                                    color = ImVec4(1.0f, 0.8f, 0.35f, 1.0f); // Yellow
                                } else if (type == "property") {
                                    item_name = "[P] " + attr;
                                    color = ImVec4(0.35f, 0.9f, 0.55f, 1.0f); // Green
                                } else {
                                    item_name = "[V] " + attr;
                                    color = ImVec4(0.9f, 0.9f, 0.9f, 1.0f); // White
                                }

                                ImGui::PushID(attr.c_str());
                                ImGui::PushStyleColor(ImGuiCol_Text, color);

                                if (ImGui::Selectable(item_name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                                    if (ImGui::IsMouseDoubleClicked(0)) {
                                        std::string api_call = mod + "." + attr + (is_fn ? "()" : "");
                                        insertTextIntoEditor(api_call);
                                    }
                                }

                                ImGui::PopStyleColor();

                                // Show interactive docstring tooltip on hover
                                if (ImGui::IsItemHovered()) {
                                    std::string doc = getAttributeDocstring(mod, attr);
                                    if (!doc.empty()) {
                                        ImGui::BeginTooltip();
                                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.35f, 1.0f), "%s.%s (%s)", mod.c_str(), attr.c_str(), type.c_str());
                                        ImGui::Separator();
                                        ImGui::TextUnformatted(doc.c_str());
                                        ImGui::EndTooltip();
                                    }
                                }
                                ImGui::PopID();
                            }
                        }
                        ImGui::TreePop();
                    }
                }
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndGroup();

    ImGui::End();
}

bool wantsInputCapture() {
    return g_wants_input_capture;
}

} // namespace rtpython
