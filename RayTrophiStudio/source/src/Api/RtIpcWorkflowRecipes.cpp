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
                {"scene.list_objects", "find the target object", "", "verify object exists", ""},
                {"fluid.create_domain", "create a gas domain around the object", "object exists", "verify gas domain bounds", "describe fluid.create_domain"},
                {"gas.set_settings", "enable fire, set ignition temperature", "gas domain exists", "verify settings applied", "describe gas.set_settings"},
                {"fluid.set_substance_material", "register the object with an MSF substance", "object exists", "", "describe fluid.set_substance_material"},
                {"flow_source.create", "add a heat source or ignition point", "gas domain exists", "", "describe flow_source.create"},
                {"timeline.set_frame", "advance time to simulate", "flow source exists", "probe viewport or get state", "describe timeline.set_frame"}
            },
            {"scene.list_objects", "fluid.create_domain", "gas.set_settings", "fluid.set_substance_material", "flow_source.create", "timeline.set_frame"}
        },
        {
            "liquid_pour",
            "Liquid Pouring Setup",
            "Create a liquid domain and an emitter to pour fluid (like water) into a container.",
            {"pour", "liquid", "water", "fluid", "fill", "container", "splash"},
            {
                {"fluid.create_domain", "create a liquid domain", "", "verify domain bounds", "describe fluid.create_domain"},
                {"flow_source.create", "create an emitter above the container", "liquid domain exists", "", "describe flow_source.create"},
                {"flow_source.update", "set the emitter to continuous flow", "emitter exists", "", "describe flow_source.update"},
                {"timeline.set_frame", "advance time to watch the fluid pour", "emitter is flowing", "probe viewport", "describe timeline.set_frame"}
            },
            {"fluid.create_domain", "flow_source.create", "flow_source.update", "timeline.set_frame"}
        },
        {
            "rigid_body_fracture",
            "Rigid Body Fracture",
            "Shatter a solid object and turn it into rigid bodies that collapse under gravity.",
            {"fracture", "shatter", "break", "collapse", "rigid", "physics", "debris"},
            {
                {"scene.list_objects", "find the object to shatter", "", "verify object exists", "describe scene.list_objects"},
                {"physics.fracture_object", "shatter into Voronoi pieces", "object exists", "verify piece count", "describe physics.fracture_object"},
                {"physics.make_fracture_group", "group pieces to stick together", "pieces exist", "", "describe physics.make_fracture_group"},
                {"physics.set_gravity", "ensure gravity is active", "", "", "describe physics.set_gravity"},
                {"timeline.set_frame", "simulate the collapse", "gravity active", "probe viewport", "describe timeline.set_frame"}
            },
            {"scene.list_objects", "physics.fracture_object", "physics.make_fracture_group", "physics.set_gravity", "timeline.set_frame"}
        },
        {
            "terrain_generation",
            "Terrain Generation with Erosion",
            "Create a new terrain heightmap, apply noise, and erode it to create natural mountains.",
            {"terrain", "mountain", "landscape", "erode", "erosion", "heightmap"},
            {
                {"terrain.create", "create a new terrain object", "", "verify terrain exists", "describe terrain.create"},
                {"terrain.apply_preset", "apply a mountain noise preset", "terrain exists", "verify preset applied", "describe terrain.apply_preset"},
                {"terrain.erode", "run hydraulic erosion", "noise applied", "inspect terrain state", "describe terrain.erode"},
                {"terrain.evaluate", "bake the terrain changes", "erosion done", "verify bake", "describe terrain.evaluate"}
            },
            {"terrain.create", "terrain.apply_preset", "terrain.erode", "terrain.evaluate"}
        },
        {
            "scatter_foliage",
            "Scatter Foliage on Surface",
            "Distribute instances of objects (like trees or rocks) over a terrain or mesh surface.",
            {"scatter", "foliage", "trees", "grass", "distribute", "instancing", "forest"},
            {
                {"scene.list_objects", "identify surface and scatter item", "", "verify both objects exist", "describe scene.list_objects"},
                {"scatter.create_group", "create a scatter group on the surface", "surface exists", "verify group created", "describe scatter.create_group"},
                {"scatter.add_source", "add the item to scatter", "group exists", "", "describe scatter.add_source"},
                {"scatter.fill", "populate the instances", "item added", "verify instance count", "describe scatter.fill"}
            },
            {"scene.list_objects", "scatter.create_group", "scatter.add_source", "scatter.fill"}
        },
        {
            "render_sequence",
            "Render Image Sequence",
            "Render an animation sequence to disk.",
            {"render", "sequence", "animation", "export", "frames", "png", "mp4"},
            {
                {"render.start_sequence", "begin background render of frames", "timeline setup complete", "verify job started", "describe render.start_sequence"},
                {"render.sequence_status", "poll to check progress", "job started", "check current frame", "describe render.sequence_status"},
                {"render.sequence_status", "wait until complete", "job running", "verify job finished", "describe render.sequence_status"}
            },
            {"render.start_sequence", "render.sequence_status", "render.cancel_sequence"}
        },
        {
            "batch_scripting",
            "Batch Scripting and Addons",
            "Write a Python script to perform heavy logical operations, mass edits, or custom pipelines, then run it inside the engine.",
            {"script", "addon", "python", "batch", "automation", "run_file", "rtpython"},
            {
                {"agent.chat_send", "plan the script logic and notify user", "", "", ""},
                {"write_to_file", "(agent tool) write the python script locally", "", "verify file exists", ""},
                {"script.run_file", "execute the script inside the engine", "script file written", "verify script results via get_scene_context or print", "describe script.run_file"}
            },
            {"script.run_file", "addons.list", "addons.enable"}
        },
        {
            "material_authoring",
            "Material Setup and Assignment",
            "Create a new material, configure properties, and assign it to an object.",
            {"material", "color", "texture", "roughness", "assign", "metallic", "shader"},
            {
                {"material.create", "create a new Principled material", "", "verify material exists", "describe material.create"},
                {"material.set", "change base_color, roughness, etc.", "material exists", "", "describe material.set"},
                {"material.set_texture", "assign texture maps", "material exists", "", "describe material.set_texture"},
                {"material.assign", "assign the material to a scene object", "material exists", "verify assignment", "describe material.assign"}
            },
            {"material.create", "material.set", "material.set_texture", "material.assign"}
        },
        {
            "lighting_setup",
            "Lighting and Atmosphere",
            "Set up environment sky and add local lights to illuminate the scene.",
            {"light", "lighting", "sun", "sky", "atmosphere", "illuminance", "point", "spot"},
            {
                {"world.set_mode", "set to 'sky'", "", "verify world mode", "describe world.set_mode"},
                {"world.set_sun_elevation", "position the sun", "mode is sky", "", "describe world.set_sun_elevation"},
                {"lights.add", "add local point or spot lights", "", "verify light added", "describe lights.add"},
                {"lights.set_intensity", "tweak light appearance", "light exists", "", "describe lights.set_intensity"}
            },
            {"world.set_mode", "world.set_sun_elevation", "lights.add", "lights.set_intensity"}
        },
        {
            "volumetric_cloud_setup",
            "Volumetric Cloud Layer",
            "Create a volumetric cloud domain driven by a noise preset.",
            {"cloud", "volume", "fog", "sky", "cumulus", "atmosphere"},
            {
                {"world.set_mode", "set background to sky", "", "verify world mode", "describe world.set_mode"},
                {"volume.create_domain", "create a generic volume box", "", "verify domain exists", "describe volume.create_domain"},
                {"volume.set_noise_preset", "apply a cloud noise pattern", "domain exists", "", "describe volume.set_noise_preset"},
                {"render.request", "update the viewport", "preset applied", "probe viewport", "describe render.request"}
            },
            {"world.set_mode", "volume.create_domain", "volume.set_noise_preset", "render.request"}
        },
        {
            "node_simulation_graph",
            "Node-based Simulation Graph",
            "Construct a node tree for advanced simulation logic.",
            {"node", "graph", "tree", "simulation", "logic", "connect", "particle"},
            {
                {"node.graph_create", "initialize a new node graph", "", "verify graph exists", "describe node.graph_create"},
                {"node.add_node", "add generator nodes", "graph exists", "verify node added", "describe node.add_node"},
                {"node.add_node", "add solver nodes", "graph exists", "verify node added", "describe node.add_node"},
                {"node.connect", "wire the nodes together via ports", "nodes exist", "verify connection", "describe node.connect"},
                {"timeline.set_frame", "advance time to evaluate the graph", "graph connected", "probe viewport", "describe timeline.set_frame"}
            },
            {"node.graph_create", "node.add_node", "node.connect", "timeline.set_frame"}
        },
        {
            "hair_grooming_physics",
            "Hair/Fur Grooming and Physics",
            "Generate a hair groom on an object and enable dynamic simulation.",
            {"hair", "fur", "groom", "physics", "strands", "wind", "simulation"},
            {
                {"scene.list_objects", "find the character or mesh", "", "verify object exists", "describe scene.list_objects"},
                {"hair.create_groom", "generate strand guides", "object exists", "verify groom exists", "describe hair.create_groom"},
                {"hair.set_physics", "enable simulation on the groom", "groom exists", "", "describe hair.set_physics"},
                {"force_field.add", "add a wind force field", "", "verify force field", "describe force_field.add"},
                {"timeline.set_frame", "step simulation", "physics enabled", "probe viewport", "describe timeline.set_frame"}
            },
            {"scene.list_objects", "hair.create_groom", "hair.set_physics", "force_field.add", "timeline.set_frame"}
        },
        {
            "multi_pass_render",
            "Multi-Pass Batch Rendering",
            "Configure render passes (AOVs) and start a batch sequence render.",
            {"render", "pass", "aov", "albedo", "depth", "normal", "sequence", "export"},
            {
                {"render.set_passes", "configure required output passes", "", "verify passes set", "describe render.set_passes"},
                {"render.start_sequence", "begin the sequence", "passes configured", "verify job started", "describe render.start_sequence"},
                {"render.sequence_status", "poll to wait for completion", "job running", "verify job finished", "describe render.sequence_status"}
            },
            {"render.set_passes", "render.start_sequence", "render.sequence_status"}
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
