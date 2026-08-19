/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcWorkflowRecipes.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#include "RtIpcWorkflowRecipes.h"
#include <algorithm>
#include <cctype>

WorkflowRecipeRegistry& WorkflowRecipeRegistry::instance() {
    static WorkflowRecipeRegistry s_instance;
    return s_instance;
}

WorkflowRecipeRegistry::WorkflowRecipeRegistry() {
    m_recipes = {
        {
            "combustion_setup",
            "Combustible Surface Setup",
            "Set up a scene object as a combustible surface that ignites, chars, and loses mass when exposed to heat.",
            {"burn", "fire", "combustion", "ignite", "wood", "char", "ash"},
            {
                "scene.list_objects -> find the target object",
                "fluid.create_domain -> create a gas domain around the object",
                "gas.set_settings -> enable fire, set ignition temperature",
                "fluid.set_substance_material -> register the object with an MSF substance (e.g. Wood)",
                "flow_source.create -> add a heat source or ignition point",
                "timeline.set_frame -> advance time to simulate"
            },
            {
                "scene.list_objects", "fluid.create_domain", "gas.set_settings",
                "fluid.set_substance_material", "flow_source.create", "timeline.set_frame"
            }
        },
        {
            "liquid_pour",
            "Liquid Pouring Setup",
            "Create a liquid domain and an emitter to pour fluid (like water) into a container.",
            {"pour", "liquid", "water", "fluid", "fill", "container", "splash"},
            {
                "fluid.create_domain -> create a liquid domain (type=fluid)",
                "flow_source.create -> create a sphere/box emitter above the container",
                "flow_source.update -> set the emitter to continuous flow",
                "timeline.set_frame -> advance time to watch the fluid pour"
            },
            {
                "fluid.create_domain", "flow_source.create", "flow_source.update", "timeline.set_frame"
            }
        },
        {
            "rigid_body_fracture",
            "Rigid Body Fracture",
            "Shatter a solid object and turn it into rigid bodies that collapse under gravity.",
            {"fracture", "shatter", "break", "collapse", "rigid", "physics", "debris"},
            {
                "scene.list_objects -> find the object to shatter",
                "physics.fracture_object -> shatter into Voronoi pieces",
                "physics.make_fracture_group -> group pieces so they stick together until impact",
                "physics.set_gravity -> ensure gravity is active",
                "timeline.set_frame -> simulate the collapse"
            },
            {
                "scene.list_objects", "physics.fracture_object", "physics.make_fracture_group",
                "physics.set_gravity", "timeline.set_frame"
            }
        },
        {
            "terrain_generation",
            "Terrain Generation with Erosion",
            "Create a new terrain heightmap, apply noise, and erode it to create natural mountains.",
            {"terrain", "mountain", "landscape", "erode", "erosion", "heightmap"},
            {
                "terrain.create -> create a new terrain object",
                "terrain.apply_preset -> apply a mountain noise preset",
                "terrain.erode -> run hydraulic erosion",
                "terrain.evaluate -> bake the terrain changes"
            },
            {
                "terrain.create", "terrain.apply_preset", "terrain.erode", "terrain.evaluate"
            }
        },
        {
            "scatter_foliage",
            "Scatter Foliage on Surface",
            "Distribute instances of objects (like trees or rocks) over a terrain or mesh surface.",
            {"scatter", "foliage", "trees", "grass", "distribute", "instancing", "forest"},
            {
                "scene.list_objects -> identify surface and scatter item",
                "scatter.create_group -> create a scatter group on the surface",
                "scatter.add_source -> add the item to scatter",
                "scatter.fill -> populate the instances based on density settings"
            },
            {
                "scene.list_objects", "scatter.create_group", "scatter.add_source", "scatter.fill"
            }
        },
        {
            "render_sequence",
            "Render Image Sequence",
            "Render an animation sequence to disk.",
            {"render", "sequence", "animation", "export", "frames", "png", "mp4"},
            {
                "render.start_sequence -> begin background render of frames",
                "render.sequence_status -> poll to check progress",
                "wait until complete"
            },
            {
                "render.start_sequence", "render.sequence_status", "render.cancel_sequence"
            }
        },
        {
            "material_authoring",
            "Material Setup and Assignment",
            "Create a new material, configure properties, and assign it to an object.",
            {"material", "color", "texture", "roughness", "assign", "metallic", "shader"},
            {
                "material.create -> create a new Principled material",
                "material.set -> change base_color, roughness, etc.",
                "material.set_texture -> (optional) assign texture maps",
                "material.assign -> assign the material to a scene object"
            },
            {
                "material.create", "material.set", "material.set_texture", "material.assign"
            }
        },
        {
            "lighting_setup",
            "Lighting and Atmosphere",
            "Set up environment sky and add local lights to illuminate the scene.",
            {"light", "lighting", "sun", "sky", "atmosphere", "illuminance", "point", "spot"},
            {
                "world.set_mode -> set to 'sky'",
                "world.set_sun_elevation -> position the sun",
                "lights.add -> add local point or spot lights",
                "lights.set_color / set_intensity -> tweak light appearance"
            },
            {
                "world.set_mode", "world.set_sun_elevation", "lights.add", "lights.set_intensity"
            }
        }
    };
}

static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

const std::vector<WorkflowRecipe>& WorkflowRecipeRegistry::all() const {
    return m_recipes;
}

const WorkflowRecipe* WorkflowRecipeRegistry::find(const std::string& id) const {
    for (const auto& recipe : m_recipes) {
        if (id == recipe.id) return &recipe;
    }
    return nullptr;
}

std::vector<const WorkflowRecipe*> WorkflowRecipeRegistry::search(const std::string& query) const {
    std::vector<std::string> query_tokens = tokenize(query);
    if (query_tokens.empty()) return {};

    struct ScoredRecipe {
        const WorkflowRecipe* recipe;
        int score;
    };
    std::vector<ScoredRecipe> results;

    for (const auto& recipe : m_recipes) {
        int score = 0;
        
        // Exact keyword matches score highest
        for (const char* kw : recipe.keywords) {
            std::string keyword(kw);
            for (const std::string& qt : query_tokens) {
                if (keyword == qt) score += 3;
            }
        }
        
        // Title/Description matches score a bit less
        std::vector<std::string> desc_tokens = tokenize(std::string(recipe.title) + " " + recipe.description);
        for (const std::string& qt : query_tokens) {
            for (const std::string& tt : desc_tokens) {
                if (tt == qt) score += 1;
            }
        }
        
        if (score > 0) {
            results.push_back({&recipe, score});
        }
    }
    
    std::sort(results.begin(), results.end(), [](const ScoredRecipe& a, const ScoredRecipe& b) {
        return a.score > b.score;
    });
    
    std::vector<const WorkflowRecipe*> final_results;
    for (const auto& sr : results) {
        final_results.push_back(sr.recipe);
    }
    
    return final_results;
}
