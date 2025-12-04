#include "scene_ui.h"
#include "imgui.h"
#include <string>
#include "scene_data.h"
#include <windows.h>
#include <commdlg.h>
#include <string>

static int new_width = image_width;
static int new_height = image_height;
static int aspect_w = 16;
static int aspect_h = 9;
static bool modelLoaded = false;
static bool loadFeedback = false; // geçici hata geri bildirimi
static float feedbackTimer = 0.0f;
inline ImVec4 ScaleColor(const ImVec4& c, float s)
{
    return ImVec4(c.x * s, c.y * s, c.z * s, c.w);
}

struct ResolutionPreset {
    const char* name;
    int w, h;
    int bw, bh;
};

static ResolutionPreset presets[] = {
    { "Custom", 0,0,0,0 },
    { "HD 720p", 1280,720, 16,9 },
    { "Full HD 1080p", 1920,1080, 16,9 },
    { "1440p", 2560,1440, 16,9 },
    { "4K UHD", 3840,2160, 16,9 },
    { "DCI 2K", 2048,1080, 19,10 },
    { "DCI 4K", 4096,2160, 19,10 },
    { "CinemaScope 4K", 4096,1716, 239,100 },
    { "Scope HD", 1920,804, 239,100 },
    { "2.35:1 HD", 1920,817, 235,100 },
    { "Vertical 1080x1920", 1080,1920, 9,16 }
};

static int preset_index = 0;
static int current_theme = 0;
static void HelpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(450.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}


std::string openFileDialogW(const wchar_t* filter = L"All Files\0*.*\0") {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrTitle = L"Select a file";
    ofn.hwndOwner = GetActiveWindow();
    if (GetOpenFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(size_needed - 1); // null terminatörü çıkar
        return utf8_path;
    }
    return "";
}


static std::string active_model_path = "No file selected yet.";

void SceneUI::drawLogConsole()
{
    if (!show_scene_log)
        return;

    // TRANSPARENT ve BAŞLIKSIZ pencere
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse;
       

    // İlk açılışta default boyut
    ImGui::SetNextWindowSize(ImVec2(450, 300), ImGuiCond_FirstUseEver);

    // Alpha düşür, paneli yarı transparan yap
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0.5f));

    if (ImGui::Begin("RayTrophi Log Panel", &show_scene_log, flags))
    {
        // Küçük font
        ImFont* smallFont = ImGui::GetIO().Fonts->Fonts[0];
        ImGui::PushFont(smallFont);

        // Sağ tıklama için popup
        if (ImGui::BeginPopupContextWindow())
        {
            if (ImGui::MenuItem("Log'u kopyala"))
            {
                std::vector<LogEntry> lines;
                g_sceneLog.getLines(lines);

                std::string total;
                for (auto& e : lines) {
                    const char* prefix =
                        (e.level == LogLevel::Info) ? "INFO" :
                        (e.level == LogLevel::Warning) ? "WARN" : "ERROR";
                    total += "[" + std::string(prefix) + "] " + e.msg + "\n";
                }
                ImGui::SetClipboardText(total.c_str());
            }

            if (ImGui::MenuItem("TXT olarak kaydet"))
            {
                std::ofstream f("RayTrophi_Log.txt");
                if (f)
                {
                    std::vector<LogEntry> lines;
                    g_sceneLog.getLines(lines);
                    for (auto& e : lines) {
                        const char* prefix =
                            (e.level == LogLevel::Info) ? "INFO" :
                            (e.level == LogLevel::Warning) ? "WARN" : "ERROR";
                        f << "[" << prefix << "] " << e.msg << "\n";
                    }
                    f.close();
                }
            }
            ImGui::EndPopup();
        }

        ImGui::Separator();

        ImGui::BeginChild("scroll", ImVec2(0, 0), false);

        static size_t lastCount = 0;
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        for (auto& e : lines)
        {
            ImVec4 color =
                (e.level == LogLevel::Info) ? ImVec4(1, 1, 1, 1) :
                (e.level == LogLevel::Warning) ? ImVec4(1, 1, 0, 1) :
                ImVec4(1, 0, 0, 1);

            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";

            ImGui::TextColored(color, "[%s] %s", prefix, e.msg.c_str());
        }

        if (lines.size() > lastCount)
            ImGui::SetScrollHereY(1.0f);

        lastCount = lines.size();

        ImGui::EndChild();
        ImGui::PopFont();
    }

    ImGui::End();
    ImGui::PopStyleColor();

    ClampWindowToDisplay();
}
void SceneUI::drawLogPanelEmbedded()
{
    ImFont* tinyFont = ImGui::GetIO().Fonts->Fonts.back();
    ImGui::PushFont(tinyFont);

    // Başlık reset zamanlayıcıları (global/statik olarak zaten tanımlı olmalı)
    if (titleChanged && ImGui::GetTime() > titleResetTime) {
        logTitle = "Scene Log";
        titleChanged = false;
    }

    // Başlık rengi varsa uygulayıp geri al
    if (titleChanged)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));

    // AllowItemOverlap ile header oluşturuyoruz — böylece aynı satırda başka butonlar çalışır
    ImGuiTreeNodeFlags hdrFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
    bool open = ImGui::CollapsingHeader(logTitle.c_str(), hdrFlags);

    if (titleChanged)
        ImGui::PopStyleColor();

    // Header ile aynı satıra Copy butonunu koy
    // İtem örtüşmesine izin verdiğimiz için buton tıklanabilir kalır
    float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SameLine(avail - 60.0f); // butonu sağa sabitliyoruz (60 px boşluk bırak)
    if (ImGui::SmallButton("Copy"))
    {
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        std::string total;
        total.reserve(lines.size() * 64);
        for (auto& e : lines) {
            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";
            total += "[" + std::string(prefix) + "] " + e.msg + "\n";
        }

        ImGui::SetClipboardText(total.c_str());

        // Başlığa kısa süreli bildirim ver
        logTitle = "Scene Log  (Copied)";
        titleResetTime = ImGui::GetTime() + 2.0f;
        titleChanged = true;
    }

    // Eğer header açıksa logu göster
    if (open)
    {
        ImGui::BeginChild("scroll_log", ImVec2(0, 150), true);

        static size_t lastCount = 0;
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        for (auto& e : lines)
        {
            ImVec4 color =
                (e.level == LogLevel::Info) ? ImVec4(1, 1, 1, 1) :
                (e.level == LogLevel::Warning) ? ImVec4(1, 1, 0, 1) :
                ImVec4(1, 0, 0, 1);

            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";

            ImGui::TextColored(color, "[%s] %s", prefix, e.msg.c_str());
        }

        if (lines.size() > lastCount)
            ImGui::SetScrollHereY(1.0f);
        lastCount = lines.size();

        ImGui::EndChild();
    }

    ImGui::PopFont();
}

void SceneUI::drawThemeSelector() {
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_Framed |
        ImGuiTreeNodeFlags_SpanFullWidth |
        //ImGuiTreeNodeFlags_DefaultOpen |
        ImGuiTreeNodeFlags_AllowItemOverlap |
        ImGuiTreeNodeFlags_FramePadding;

    if (ImGui::TreeNodeEx("Interface / Theme", flags))
    {

        const char* themes[] = {
         "Dark", "Light", "Classic",
         "Blender Dark",
         "RayTrophi Pro Dark",
         "Neon Cyber",
         "High Contrast"
        };


        if (ImGui::Combo("Select Theme", &current_theme, themes, IM_ARRAYSIZE(themes))) {
            ImGuiStyle& style = ImGui::GetStyle();

            // Önce full reset
            style = ImGuiStyle();
            switch (current_theme) {
            case 0: ImGui::StyleColorsDark(); break;
            case 1: ImGui::StyleColorsLight(); break;
            case 2: ImGui::StyleColorsClassic(); break;
            case 3: {
                ImGuiStyle& style = ImGui::GetStyle();
                ImVec4* colors = style.Colors;

                style.FrameRounding = 3.0f;
                style.WindowRounding = 2.0f;
                style.ScrollbarRounding = 3.0f;
                style.GrabRounding = 3.0f;
                style.PopupRounding = 2.0f;
                style.TabRounding = 3.0f;

                // Arka planlar
                colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, panel_alpha);
                colors[ImGuiCol_ChildBg] = ImVec4(0.10f, 0.10f, 0.12f, 0.90f);
                colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.14f, 0.95f);

                // Metin
                colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
                colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);

                // Frame’ler
                colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
                colors[ImGuiCol_FrameBgHovered] = ImVec4(0.25f, 0.25f, 0.28f, 1.00f);
                colors[ImGuiCol_FrameBgActive] = ImVec4(0.30f, 0.30f, 0.33f, 1.00f);

                // Butonlar
                colors[ImGuiCol_Button] = ImVec4(0.20f, 0.45f, 0.85f, 0.80f);
                colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.55f, 0.95f, 1.00f);
                colors[ImGuiCol_ButtonActive] = ImVec4(0.18f, 0.45f, 0.85f, 1.00f);

                // Header (Tree, Collapsing)
                colors[ImGuiCol_Header] = ImVec4(0.22f, 0.24f, 0.28f, 1.00f);
                colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.30f, 0.34f, 1.00f);
                colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.32f, 0.36f, 1.00f);

                // Scrollbar
                colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.10f, 0.11f, 0.60f);
                colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.28f, 0.28f, 0.33f, 1.00f);
                colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.36f, 0.36f, 0.41f, 1.00f);
                colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.50f, 0.50f, 0.55f, 1.00f);

                // Tabs (Render settings’de çok kullanıyorsun)
                colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.17f, 0.20f, 1.00f);
                colors[ImGuiCol_TabHovered] = ImVec4(0.28f, 0.30f, 0.36f, 1.00f);
                colors[ImGuiCol_TabActive] = ImVec4(0.22f, 0.24f, 0.28f, 1.00f);

                // Border
                colors[ImGuiCol_Border] = ImVec4(0.05f, 0.05f, 0.05f, 0.50f);
                colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

                // Slider / Grab
                colors[ImGuiCol_SliderGrab] = ImVec4(0.28f, 0.56f, 1.0f, 0.9f);
                colors[ImGuiCol_SliderGrabActive] = ImVec4(0.35f, 0.63f, 1.0f, 1.0f);
                break;
            }
            case 4: {
                ImGuiStyle& style = ImGui::GetStyle();
                ImVec4* c = style.Colors;

                style.FrameRounding = 3.0f;
                style.WindowRounding = 2.0f;
                style.ScrollbarRounding = 3.0f;
                style.GrabRounding = 3.0f;
                style.PopupRounding = 2.0f;
                style.TabRounding = 3.0f;

                c[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.11f, 0.12f, panel_alpha);
                c[ImGuiCol_ChildBg] = ImVec4(0.11f, 0.11f, 0.12f, 0.95f);
                c[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.14f, 1.0f);

                c[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 0.95f, 1);
                c[ImGuiCol_TextDisabled] = ImVec4(0.45f, 0.45f, 0.45f, 1);

                c[ImGuiCol_FrameBg] = ImVec4(0.18f, 0.18f, 0.19f, 1);
                c[ImGuiCol_FrameBgHovered] = ImVec4(0.25f, 0.25f, 0.27f, 1);
                c[ImGuiCol_FrameBgActive] = ImVec4(0.30f, 0.30f, 0.33f, 1);

                c[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.27f, 1);
                c[ImGuiCol_ButtonHovered] = ImVec4(0.35f, 0.35f, 0.38f, 1);
                c[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.20f, 0.22f, 1);

                c[ImGuiCol_Header] = ImVec4(0.22f, 0.22f, 0.24f, 1);
                c[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.28f, 0.30f, 1);
                c[ImGuiCol_HeaderActive] = ImVec4(0.32f, 0.32f, 0.34f, 1);

                c[ImGuiCol_SliderGrab] = ImVec4(0.4f, 0.4f, 0.45f, 1);
                c[ImGuiCol_SliderGrabActive] = ImVec4(0.55f, 0.55f, 0.60f, 1);

                c[ImGuiCol_Border] = ImVec4(0, 0, 0, 0.4f);

                break;
            }
            case 5: {
                ImGuiStyle& style = ImGui::GetStyle();
                ImVec4* c = style.Colors;

                style.WindowRounding = 4;
                style.FrameRounding = 4;
                style.GrabRounding = 4;

                c[ImGuiCol_WindowBg] = ImVec4(0.02f, 0.02f, 0.04f, panel_alpha);
                c[ImGuiCol_Text] = ImVec4(0.65f, 1.0f, 0.65f, 1);

                c[ImGuiCol_Button] = ImVec4(0.05f, 0.25f, 0.05f, 1);
                c[ImGuiCol_ButtonHovered] = ImVec4(0.10f, 0.40f, 0.10f, 1);
                c[ImGuiCol_ButtonActive] = ImVec4(0.05f, 0.30f, 0.05f, 1);

                c[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.13f, 0.05f, 1);
                c[ImGuiCol_FrameBgHovered] = ImVec4(0.08f, 0.18f, 0.08f, 1);
                c[ImGuiCol_FrameBgActive] = ImVec4(0.10f, 0.25f, 0.10f, 1);

                c[ImGuiCol_Header] = ImVec4(0.10f, 0.25f, 0.10f, 1);
                c[ImGuiCol_HeaderHovered] = ImVec4(0.15f, 0.35f, 0.15f, 1);
                c[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.45f, 0.20f, 1);

                c[ImGuiCol_SliderGrab] = ImVec4(0.40f, 1.0f, 0.40f, 1);
                c[ImGuiCol_SliderGrabActive] = ImVec4(0.60f, 1.0f, 0.60f, 1);

                c[ImGuiCol_Tab] = ImVec4(0.05f, 0.15f, 0.05f, 1);
                c[ImGuiCol_TabActive] = ImVec4(0.10f, 0.25f, 0.10f, 1);

                c[ImGuiCol_Border] = ImVec4(0.0f, 1.0f, 0.0f, 0.2f);
                break;
            }
            case 6: {
                ImGuiStyle& style = ImGui::GetStyle();
                ImVec4* c = style.Colors;

                style.WindowRounding = 0;
                style.FrameRounding = 0;
                style.GrabRounding = 0;

                c[ImGuiCol_WindowBg] = ImVec4(0.0f, 0.0f, 0.0f, panel_alpha);
                c[ImGuiCol_Text] = ImVec4(1, 1, 1, 1);
                c[ImGuiCol_TextDisabled] = ImVec4(0.6f, 0.6f, 0.6f, 1);

                c[ImGuiCol_Button] = ImVec4(0.0f, 0.0f, 0.0f, 1);
                c[ImGuiCol_ButtonHovered] = ImVec4(0.2f, 0.2f, 0.2f, 1);
                c[ImGuiCol_ButtonActive] = ImVec4(0.35f, 0.35f, 0.35f, 1);

                c[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.05f, 1);
                c[ImGuiCol_FrameBgHovered] = ImVec4(0.15f, 0.15f, 0.15f, 1);
                c[ImGuiCol_FrameBgActive] = ImVec4(0.25f, 0.25f, 0.25f, 1);

                c[ImGuiCol_Border] = ImVec4(1, 1, 1, 0.4f);
                c[ImGuiCol_ScrollbarGrab] = ImVec4(1, 1, 1, 0.5f);
                c[ImGuiCol_ScrollbarGrabActive] = ImVec4(1, 1, 1, 1);

                break;
            }


            }
        }
        // Panel Transparency   
        if (ImGui::SliderFloat("Panel Transparency", &panel_alpha, 0.1f, 1.0f, "%.2f")) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.Colors[ImGuiCol_WindowBg].w = panel_alpha;
        }

        ImGui::TreePop();
    }
}
void SceneUI::drawResolutionPanel()
{
    // Resolution Panel Collapsible
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_Framed |
        ImGuiTreeNodeFlags_SpanFullWidth |
        ImGuiTreeNodeFlags_AllowItemOverlap |
        ImGuiTreeNodeFlags_FramePadding;

    if (ImGui::TreeNodeEx("Resolution Settings", flags))
    {
        ImGui::Text("Preset Resolution");
        if (ImGui::Combo("Presets", &preset_index,
            [](void* data, int idx, const char** out_text) {
                *out_text = ((ResolutionPreset*)data)[idx].name;
                return true;
            }, presets, IM_ARRAYSIZE(presets)))
        {
            if (preset_index != 0) {
                new_width = presets[preset_index].w;
                new_height = presets[preset_index].h;
                aspect_w = presets[preset_index].bw;
                aspect_h = presets[preset_index].bh;
            }
        }

        ImGui::Spacing();
        ImGui::Text("Resolution");
        ImGui::InputInt("Width", &new_width);
        ImGui::InputInt("Height", &new_height);
        ImGui::InputInt("Aspect W", &aspect_w);
        ImGui::SameLine();
        ImGui::InputInt("Aspect H", &aspect_h);

        bool resolution_changed =
            (new_width != last_applied_width) ||
            (new_height != last_applied_height) ||
            (aspect_w != last_applied_aspect_w) ||
            (aspect_h != last_applied_aspect_h);

        ImGui::BeginDisabled(!resolution_changed);

        if (ImGui::Button("Apply"))
        {
            float ar = aspect_h ? float(aspect_w) / aspect_h : 1.0f;
            pending_aspect_ratio = ar;
            pending_width = new_width;
            pending_height = new_height;
            aspect_ratio = ar;
            pending_resolution_change = true;

            last_applied_width = new_width;
            last_applied_height = new_height;
            last_applied_aspect_w = aspect_w;
            last_applied_aspect_h = aspect_h;
        }

        ImGui::EndDisabled();

        ImGui::TreePop();
    }

}

void SceneUI::drawToneMapPanel(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    ImGui::SetNextWindowSize(ImVec2(340, 280), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(700, screen_y - 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Tone Mapping", nullptr);

    ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.6f, 1), "Post-Processing Controls");
    ImGui::Separator();

    // -------- Main Parameters --------
    if (ImGui::TreeNodeEx("Main Post-Processing")) {
        ImGui::SliderFloat("Gamma", &ctx.color_processor.params.global_gamma, 0.5f, 3.0f, "%.2f");
        ImGui::SliderFloat("Exposure", &ctx.color_processor.params.global_exposure, 0.1f, 5.0f, "%.2f");
        ImGui::SliderFloat("Saturation", &ctx.color_processor.params.saturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Temperature (K)", &ctx.color_processor.params.color_temperature, 1000.0f, 10000.0f, "%.0f");
        ImGui::TreePop();
    }

    // -------- Tonemapping Type --------
    if (ImGui::TreeNodeEx("Tonemapping Type")) {
        const char* tone_names[] = { "AGX", "ACES", "Uncharted", "Filmic", "None" };
        int selected_tone = static_cast<int>(ctx.color_processor.params.tone_mapping_type);
        if (ImGui::Combo("Tonemapping", &selected_tone, tone_names, IM_ARRAYSIZE(tone_names))) {
            ctx.color_processor.params.tone_mapping_type = static_cast<ToneMappingType>(selected_tone);
        }
        ImGui::TreePop();
    }

    // -------- Effects --------
    if (ImGui::TreeNodeEx("Effects")) {
        ImGui::Checkbox("Vignette", &ctx.color_processor.params.enable_vignette);
        if (ctx.color_processor.params.enable_vignette) {
            ImGui::SliderFloat("Vignette Strength", &ctx.color_processor.params.vignette_strength, 0.0f, 2.0f, "%.2f");
        }
        ImGui::TreePop();
    }

    // -------- Actions --------
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::Button("Apply Tonemap", ImVec2(110, 0))) ctx.apply_tonemap = true;
    ImGui::SameLine();
    if (ImGui::Button("Reset", ImVec2(110, 0))) ctx.reset_tonemap = true;

    ImGui::End();
    ClampWindowToDisplay();
}

void SceneUI::drawCameraPanel(UIContext& ctx, float screen_y)
{
    ImGui::SetNextWindowSize(ImVec2(340, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, screen_y - 320), ImGuiCond_FirstUseEver);
    ImGui::Begin("Camera", nullptr);
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_Framed |
        ImGuiTreeNodeFlags_SpanFullWidth |
        //ImGuiTreeNodeFlags_DefaultOpen |
        ImGuiTreeNodeFlags_AllowItemOverlap |
        ImGuiTreeNodeFlags_FramePadding;
    if (!ctx.scene.camera) {
        ImGui::Text("No Camera");
        ImGui::End();
        return;
    }

    Vec3 pos = ctx.scene.camera->lookfrom;
    Vec3 target = ctx.scene.camera->lookat;
    float fov = (float)ctx.scene.camera->vfov;
    float& aperture = ctx.scene.camera->aperture;
    float& focus_dist = ctx.scene.camera->focus_dist;

    // -------- Position & Target --------
    static bool targetLock = true;
    if (ImGui::TreeNodeEx("Position & Target", flags)) {
        ImGui::PushItemWidth(300);
        bool pos_changed = ImGui::DragFloat3("Position", &pos.x, 0.01f);

        bool target_changed = false;
        if (!targetLock)
            target_changed = ImGui::DragFloat3("Target", &target.x, 0.01f);
        else
            ImGui::BeginDisabled(), ImGui::DragFloat3("Target", &target.x, 0.1f), ImGui::EndDisabled();

        ImGui::PopItemWidth();
        ImGui::Checkbox("Lock Target", &targetLock);

        if (pos_changed) {
            if (targetLock)
                ctx.scene.camera->moveToTargetLocked(pos);
            else {
                ctx.scene.camera->lookfrom = pos;
                ctx.scene.camera->origin = pos;
                ctx.scene.camera->update_camera_vectors();
            }
        }
        if (target_changed) {
            ctx.scene.camera->lookat = target;
            ctx.scene.camera->update_camera_vectors();
        }
        ImGui::TreePop();
    }

    // -------- View Parameters --------
    if (ImGui::TreeNodeEx("View Parameters", flags)) {
        if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
            ctx.scene.camera->vfov = fov;
            ctx.scene.camera->fov = fov;
            ctx.scene.camera->update_camera_vectors();
        }
        HelpMarker("Field of View");
        ImGui::TreePop();
    }

    // -------- Depth of Field --------
    if (ImGui::TreeNodeEx("Depth of Field & Bokeh", flags)) {
        bool aperture_changed = ImGui::SliderFloat("Aperture", &aperture, 0.0f, 5.0f, "%.2f");
        bool focus_changed = ImGui::DragFloat("Focus Distance", &focus_dist, 0.05f, 0.01f, 100.0f);

        if (aperture_changed || focus_changed) {
            ctx.scene.camera->lens_radius = aperture * 0.5f;
            ctx.scene.camera->update_camera_vectors();
        }

        ImGui::SliderInt("Blade Count", &ctx.scene.camera->blade_count, 3, 12);
        ImGui::TreePop();
    }


    // -------- Mouse Control --------
    if (ImGui::TreeNodeEx("Mouse Control", flags)) {
        ImGui::Checkbox("Enable Mouse Look", &ctx.mouse_control_enabled);
        if (ctx.mouse_control_enabled)
            ImGui::SliderFloat("Sensitivity", &ctx.mouse_sensitivity, 0.01f, 0.5f, "%.3f");
        ImGui::TreePop();
    }

    ImGui::End();
    ClampWindowToDisplay();
}

void SceneUI::drawLightsPanel(UIContext& ctx, float screen_y)
{
    ImGui::SetNextWindowSize(ImVec2(340, 260), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(340, screen_y - 280), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lights", nullptr);

    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1), "Scene Lights");

    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto light = ctx.scene.lights[i];
        if (!light) continue;

        std::string label = "Light #" + std::to_string(i);

        if (ImGui::TreeNode(label.c_str())) {
            const char* names[] = { "Point", "Directional", "Spot", "Area" };
            int index = (int)light->type();
            if (index >= 0 && index < 4)
                ImGui::Text("Type: %s", names[index]);
            else
                ImGui::Text("Type: Unknown");

            ImGui::DragFloat3("Position", &light->position.x, 0.1f);

            if (light->type() == LightType::Directional || light->type() == LightType::Spot)
                ImGui::DragFloat3("Direction", &light->direction.x, 0.1f);

            ImGui::ColorEdit3("Color", &light->color.x);
            ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0, 1000.0f);

            if (light->type() == LightType::Point ||
                light->type() == LightType::Area ||
                light->type() == LightType::Directional)
                ImGui::DragFloat("Radius", &light->radius, 0.01f, 0.01f, 100.0f);

            ImGui::TreePop();
        }
    }   
    ImGui::End();
    ClampWindowToDisplay();
}
void SceneUI::drawRenderSettingsPanel(UIContext& ctx, float screen_y)
{
    ImGui::SetNextWindowSize(ImVec2(360, 360), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1040, screen_y - 280), ImGuiCond_FirstUseEver);
    ImGui::Begin("Render Settings", nullptr);   
    drawThemeSelector();
    drawResolutionPanel();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Model & Scene");
    bool disabled = scene_loading.load();
    if (disabled)
        ImGui::BeginDisabled();

    // Load Model Button and Log Checkbox on same line

   // Model yüklenmiş mi?
    bool loaded = (ctx.scene.initialized);

    // Renkleri seç
    ImVec4 baseColor = loaded
        ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)   // yeşil buton
        : ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // kırmızı buton

    // Yazı rengi (kontrast)
    ImVec4 textColor = loaded
        ? ImVec4(0.0f, 0.0f, 0.0f, 1.0f)   // yeşile siyah yazı
        : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);  // kırmızıya beyaz yazı

    // Style push
    ImGui::PushStyleColor(ImGuiCol_Button, baseColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ScaleColor(baseColor, 1.2f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ScaleColor(baseColor, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_Text, textColor);

    // Buton text’i
    const char* label = loaded ? "Loaded " : "Load Model";

    if (ImGui::Button(label, ImVec2(100, 0)))
    {
#ifdef _WIN32
        std::string file = openFileDialogW(
            L"3D Files\0*.gltf;*.glb;*.fbx;*.obj;*.dae;*.3ds;*.blend;*.ply;*.stl\0All Files\0*.*\0"
        );

        if (!file.empty()) {
            // normal yükleme işlerin…
            scene_loading = true;
            scene_loading_done = false;
            active_model_path = file;

            std::thread loader_thread([this, file, &ctx]() {
                SCENE_LOG_INFO("Starting async scene load...");
                ctx.scene.clear();
                ctx.renderer.create_scene(ctx.scene, ctx.optix_gpu_ptr, file);
                if (ctx.scene.camera)
                    ctx.scene.camera->update_camera_vectors();
                ctx.render_settings.start_animation_render = false;
                ctx.start_render = false;

                SCENE_LOG_INFO("Scene loaded successfully.");
                scene_loading = false;
                scene_loading_done = true;
                });

            loader_thread.detach();
        }
#endif
    }

    // Style pop
    ImGui::PopStyleColor(4);

    
    if (disabled)
        ImGui::EndDisabled();
   
    ImGui::SameLine();
    ImGui::TextWrapped("Model: %s", active_model_path.c_str());
    ImGui::Separator();
    ImGui::PushItemWidth(180);
    ImGui::Separator();
    ImGui::PushItemWidth(180);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Render Engine");


    if (ctx.scene.initialized) {
        // --- GPU Section ---
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1), "GPU (OptiX)");
        bool prev_use_optix = ctx.render_settings.use_optix;
        if (!g_hasOptix) {
            ImGui::BeginDisabled();
            ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "(No RTX GPU detected)");
            ImGui::EndDisabled();
        }
        else if (ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix)) {
            if (ctx.render_settings.use_optix != prev_use_optix) {
                SCENE_LOG_INFO(ctx.render_settings.use_optix ? "OptiX enabled" : "OptiX disabled");
            }
        }
        ImGui::SameLine();
        HelpMarker("Enables GPU acceleration via NVIDIA OptiX. Requires an RTX-class GPU.");

        // --- CPU Section ---
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "CPU (BVH)");
        const char* bvh_options[] = { "Embree", "In-house" };
        static int current_bvh = ctx.render_settings.UI_use_embree ? 0 : 1;
        if (ImGui::Combo("BVH Type", &current_bvh, bvh_options, IM_ARRAYSIZE(bvh_options))) {
            ctx.render_settings.UI_use_embree = (current_bvh == 0);
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        }
        if (!ctx.render_settings.UI_use_embree) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "(Experimental)");
        }
        ImGui::SameLine();
        HelpMarker("Select which BVH structure to use for acceleration. Embree = highly optimized, In-house = custom implementation (Experimental).");

        // --- Denoiser Section ---
        ImGui::TextColored(ImVec4(0.8f, 1.0f, 0.6f, 1), "Denoiser");
        bool prev_use_denoiser = ctx.render_settings.use_denoiser;
        if (ImGui::Checkbox("Use Denoiser", &ctx.render_settings.use_denoiser)) {
            if (ctx.render_settings.use_denoiser != prev_use_denoiser) {
                SCENE_LOG_INFO(ctx.render_settings.use_denoiser ? "Denoiser enabled" : "Denoiser disabled");
            }
        }
        ImGui::SameLine();
        HelpMarker("Applies denoising to reduce noise after rendering. Based on Intel OIDN.");
        if (ctx.render_settings.use_denoiser) {
            ImGui::SliderFloat("Denoiser Blend", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f, "%.2f");
            ImGui::SameLine();
            HelpMarker("Blends the denoised result with the original. 1 = fully denoised, 0 = original image.");
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Adaptive Sampling");

        // Min Samples
        if (ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples, 1.0f, 1, 1024)) {
            SCENE_LOG_INFO("Min Samples changed to " + std::to_string(ctx.render_settings.min_samples));
        }
        ImGui::SameLine(); HelpMarker("Minimum number of samples per pixel before noise variance is evaluated.");

        // Max Samples
        if (ImGui::DragInt("Max Samples", &ctx.render_settings.max_samples, 1.0f, 1, 2048)) {
            SCENE_LOG_INFO("Max Samples changed to " + std::to_string(ctx.render_settings.max_samples));
        }
        ImGui::SameLine(); HelpMarker("Maximum number of samples per pixel. Higher values allow cleaner results but take longer.");

        // Variance Threshold
        if (ImGui::SliderFloat("Variance Threshold", &ctx.render_settings.variance_threshold, 0.001f, 1.0f, "%.5f")) {
            SCENE_LOG_INFO("Variance Threshold changed to " + std::to_string(ctx.render_settings.variance_threshold));
        }
        ImGui::SameLine(); HelpMarker("Pixels with variance below this threshold will stop sampling early. Lower = cleaner but slower.");

        // Max Bounces
        ImGui::Separator();
        if (ImGui::DragInt("Max Bounce", &ctx.render_settings.max_bounces, 1.0f, 1, 32)) {
            SCENE_LOG_INFO("Max Bounce changed to " + std::to_string(ctx.render_settings.max_bounces));
        }
        ImGui::SameLine(); HelpMarker("Maximum number of ray bounces. Higher values improve indirect lighting, reflections, and refractions but increase render time.");

        // Environment
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Environment");
        if (ImGui::ColorEdit3("Background Color", &ctx.scene.background_color.x)) {
            SCENE_LOG_INFO("Background Color changed to RGB(" +
                std::to_string(int(ctx.scene.background_color.x * 255)) + ", " +
                std::to_string(int(ctx.scene.background_color.y * 255)) + ", " +
                std::to_string(int(ctx.scene.background_color.z * 255)) + ")");
        }

        ImGui::Separator();
        if (ImGui::Button("Start Render", ImVec2(150, 0))) {
            ctx.start_render = true;
        }
    }
    else {
        ImGui::BeginDisabled();
        ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix);
        ImGui::Checkbox("Use Denoiser", &ctx.render_settings.use_denoiser);
        ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples);
        ImGui::DragInt("Max Samples", &ctx.render_settings.max_samples);
        ImGui::SliderFloat("Variance Threshold", &ctx.render_settings.variance_threshold, 0.001f, 1.0f, "%.5f");
        ImGui::DragInt("Max Bounce", &ctx.render_settings.max_bounces, 1.0f, 1, 32);
        ImGui::ColorEdit3("Background Color", &ctx.scene.background_color.x);
        ImGui::Button("Start Render", ImVec2(150, 0));
        ImGui::EndDisabled();
    }


    ImGui::SameLine();
    ImGui::BeginDisabled();
    if (ImGui::Button("Stop", ImVec2(150, 0))) ctx.start_render = false;
    ImGui::EndDisabled();
    ImGui::Separator();
    // Show current render time text
    ImGui::Text("Last Render Time: %.4f sec", last_render_time_ms);
    // Clamp to [0, 1] range for progress bar
    float normalized = std::fmin(last_render_time_ms / 1000.0f, 1.0f); // 1 saniyeyi 100%
    // Label inside progress bar
    char label1[64];
    snprintf(label1, sizeof(label), "%.4f sec", last_render_time_ms);
    // Bar visualization
    ImGui::ProgressBar(normalized, ImVec2(-1, 25), label1);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1), "Animation");

    drawAnimationSettings(ctx);

    ImGui::Spacing();
   
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Save Image");
    if (ImGui::Button("Save Image As...", ImVec2(310, 0))) {
        ctx.render_settings.save_image_requested = true;
    }
    // Show current render time text   
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();
    drawLogPanelEmbedded(); // *** burada çağırıyoruz ***
    ImGui::End();
    ClampWindowToDisplay();
}
// çağır: ImGui::Begin("Camera"); ... ClampWindowToDisplay(); ImGui::End();

void SceneUI::ClampWindowToDisplay()
{
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 disp = io.DisplaySize;

    ImVec2 win_pos = ImGui::GetWindowPos();
    ImVec2 win_size = ImGui::GetWindowSize();

    // Eğer pencere invisible veya 0 boyutluysa çık
    if (win_size.x <= 0.0f || win_size.y <= 0.0f) return;

    float x = win_pos.x;
    float y = win_pos.y;

    // Sağ/bottom taşmaları düzelt
    if (x + win_size.x > disp.x) x = disp.x - win_size.x;
    if (y + win_size.y > disp.y) y = disp.y - win_size.y;

    // Negatif değerlere izin verme
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;

    // Pozisyon değiştiyse uygula
    if (x != win_pos.x || y != win_pos.y) {
        ImGui::SetWindowPos(ImVec2(x, y), ImGuiCond_Always);
    }

    // Eğer pencere ekran boyutuna göre taşarsa, boyutu da düzelt
    bool size_changed = false;
    float new_width = win_size.x;
    float new_height = win_size.y;

    if (win_size.x > disp.x) { new_width = disp.x; size_changed = true; }
    if (win_size.y > disp.y) { new_height = disp.y; size_changed = true; }

    if (size_changed) {
        ImGui::SetWindowSize(ImVec2(new_width, new_height), ImGuiCond_Always);
    }
}
void SceneUI::drawAnimationSettings(UIContext& ctx)
{
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_Framed |
        ImGuiTreeNodeFlags_SpanFullWidth |
        //ImGuiTreeNodeFlags_DefaultOpen |
        ImGuiTreeNodeFlags_AllowItemOverlap |
        ImGuiTreeNodeFlags_FramePadding;
    // Sadece bu node'un text rengini kırmızı yap
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));

    bool open = ImGui::TreeNodeEx("Animation Rendering (Experimental)", flags);

    ImGui::PopStyleColor();
    if (open)
    {
       
        ImGui::SliderFloat("Duration (sec)",
            &ctx.render_settings.animation_duration,
            0.1f, 60.0f, "%.1f");
        HelpMarker("Length of the animation in seconds.");

        ImGui::SliderInt("FPS",
            &ctx.render_settings.animation_fps,
            1, 60);
        HelpMarker("Frames per second for animation rendering.");

        ImGui::Spacing();

        if (ImGui::Button("Start CPU Animation Render", ImVec2(-1, 0)))
        {
            ctx.render_settings.start_animation_render = true;
            ctx.start_render = true;
        }

        ImGui::TreePop();
    }
}

void SceneUI::draw(UIContext& ctx)
{
    
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);   
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, panel_alpha));

    drawCameraPanel(ctx, screen_y);
    drawLightsPanel(ctx, screen_y);
    drawToneMapPanel(ctx);
    drawRenderSettingsPanel(ctx, screen_y);
    //drawLogConsole();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}


