#ifndef SCENE_UI_HELP_HPP
#define SCENE_UI_HELP_HPP

#include "scene_ui.h"
#include "imgui.h"
#include "ui_modern.h"
#include <vector>
#include <string>
#include <map>

struct HelpSection {
    std::string title;
    std::string content;
    std::string image_desc; // Placeholder description for images
};

struct HelpTopic {
    std::string id;
    std::string title;
    std::string icon; // Ex: "[Water]"
    std::vector<HelpSection> sections;
};

class HelpSystem {
public:
    static HelpSystem& instance() {
        static HelpSystem inst;
        return inst;
    }

    void draw(bool* p_open) {
        if (!*p_open) return;

        ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Documentation & Help", p_open, ImGuiWindowFlags_NoCollapse)) {
            
            // Layout: Sidebar (Topics) | Content (Scrollable)
            
            static float sidebar_width = 250.0f;
            
            // Sidebar
            ImGui::BeginChild("HelpSidebar", ImVec2(sidebar_width, 0), true);
            
            UIWidgets::ColoredHeader("Manual Topics", ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::Spacing();
            
            for (size_t i = 0; i < topics.size(); i++) {
                bool selected = (selected_topic_idx == i);
                std::string label = topics[i].icon + " " + topics[i].title;
                
                if (selected) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.9f, 0.4f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2f, 0.3f, 0.4f, 1.0f));
                }
                
                if (ImGui::Selectable(label.c_str(), selected)) {
                    selected_topic_idx = (int)i;
                }
                
                if (selected) {
                    ImGui::PopStyleColor(2);
                }
            }
            ImGui::EndChild();
            
            ImGui::SameLine();
            
            // Content
            ImGui::BeginChild("HelpContent", ImVec2(0, 0), true);
            
            if (selected_topic_idx >= 0 && selected_topic_idx < topics.size()) {
                const auto& topic = topics[selected_topic_idx];
                
                // Title
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Default font (bold usually)
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", topic.title.c_str());
                ImGui::Separator();
                ImGui::PopFont();
                ImGui::Spacing();
                
                for (const auto& sec : topic.sections) {
                    // Section Title
                    if (!sec.title.empty()) {
                        ImGui::Spacing();
                        UIWidgets::ColoredHeader(sec.title.c_str(), ImVec4(0.7f, 0.9f, 0.6f, 1.0f));
                    }
                    
                    // Image Placeholder (Simulating PDF images)
                    if (!sec.image_desc.empty()) {
                        float avail_w = ImGui::GetContentRegionAvail().x;
                        float img_h = 200.0f;
                        ImVec2 p = ImGui::GetCursorScreenPos();
                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        
                        // Background
                        dl->AddRectFilled(p, ImVec2(p.x + avail_w, p.y + img_h), IM_COL32(30, 35, 40, 255));
                        dl->AddRect(p, ImVec2(p.x + avail_w, p.y + img_h), IM_COL32(100, 100, 100, 100));
                        
                        // Text
                        std::string txt = "[ IMAGE: " + sec.image_desc + " ]";
                        ImVec2 txt_sz = ImGui::CalcTextSize(txt.c_str());
                        dl->AddText(ImVec2(p.x + (avail_w - txt_sz.x)*0.5f, p.y + (img_h - txt_sz.y)*0.5f), 
                                    IM_COL32(150, 150, 150, 255), txt.c_str());
                                    
                        ImGui::Dummy(ImVec2(avail_w, img_h));
                        ImGui::Spacing();
                    }
                    
                    // Text Content
                    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x - 20.0f);
                    ImGui::TextUnformatted(sec.content.c_str());
                    ImGui::PopTextWrapPos();
                    
                    ImGui::Spacing();
                }
            }
            
            ImGui::EndChild();
        }
        ImGui::End();
    }

private:
    std::vector<HelpTopic> topics;
    int selected_topic_idx = 0;

    HelpSystem() {
        // 1. INTRODUCTION
        topics.push_back({
            "intro", "Introduction", "[i]",
            {
                { "Welcome to RayTrophi", "RayTrophi is an advanced path tracing engine with a modular UI system. This documentation covers the new features including the Water System, Terrain Editor, and Animation Graph.", "" },
                { "Navigation", "Use Right Mouse Button to rotate the camera.\nUse Middle Mouse Button to pan.\nUse Scroll Wheel to zoom.", "Viewport navigation showing mouse inputs" }
            }
        });

        // 2. WATER SYSTEM
        topics.push_back({
            "water", "Water System", "[~]",
            {
                { "Overview", "The Water System allows you to create realistic oceans, lakes, and rivers using physically based parameters and FFT simulations.", "" },
                { "FFT Ocean (Tessendorf)", "Enable 'FFT Ocean' in the Water Panel to use film-quality wave simulations.\n\n- Wind Speed: Controls wave energy (>30m/s is storm).\n- Choppiness: Makes wave peaks sharper.\n- Ocean Size: Determines the world-space size of the simulation tile.", "FFT Ocean settings panel" },
                { "Physics & Looks", "Adjust 'Absorption Density' to control how murky the water looks. 'Deep Color' and 'Shallow Color' control the gradient based on depth.", "Water color and absorption examples" }
            }
        });

        // 3. RIVER SYSTEM
        topics.push_back({
            "river", "River System", "[S]",
            {
                { "Spline Editing", "Rivers are created using cubic bezier splines. Click 'Add Point' to extend the river. Select points to move them with Gizmos.", "River spline with control points" },
                { "Flow Physics", "The river automatically calculates flow vectors. You can adjust 'Turbulence', 'Flow Speed', and 'Banking' (tilt on curves) in the Physics panel.", "" }
            }
        });

        // 4. TERRAIN SYSTEM
        topics.push_back({
            "terrain", "Terrain & Erosion", "[M]",
            {
                { "Hydraulic Erosion", "The terrain editor features a particle-based hydraulic erosion system. It simulates rain falling on the terrain, dissolving soil, and depositing it elsewhere.\n\nParameters:\n- Rain Amount: Number of droplets.\n- Solubility: How easily soil dissolves.\n- Evaporation: How fast water dries.", "Erosion simulation steps" },
                { "Node Graph", "For advanced terrain generation, use the Node Graph editor to mix noises (Perlin, Worley) and masks.", "Terrain Node Graph interface" }
            }
        });

        // 5. ANIMATION
        topics.push_back({
            "anim", "Animation System", "[A]",
            {
                { "Keyframing", "Most sliders in the UI have a diamond icon. Click it to add a keyframe at the current timeline position. The timeline at the bottom shows all keyframes.", "Slider with keyframe diamond active" },
                { "Animation Node Graph", "Create complex character behaviors using the Animation Graph. Connect State nodes and Blend Trees to drive animations.", "Animation Graph with state nodes" }
            }
        });
        
        // 6. PRO CAMERA
        topics.push_back({
            "cam", "Pro Camera", "[O]",
            {
                { "Physical Lens", "The camera simulates a physical lens.\n- Aperture (f-stop): Controls Depth of Field. Lower values = blurrier background.\n- Focal Distance: The distance where objects are sharp.", "" },
                { "Focus Assist", "Enable 'Show Focus Ring' in Viewport Overlay to see exactly where the focus plane is in 3D space.", "Viewport with red focus ring" }
            }
        });
    }
};

#endif
