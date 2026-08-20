/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcWorkflowRecipes.h
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#pragma once

#include <string>
#include <vector>

struct WorkflowStep {
    const char* action;
    const char* purpose;
    const char* requires_state;
    const char* verify;
    const char* on_failure;
};

struct WorkflowRecipe {
    const char* id;
    const char* title;
    const char* description;
    
    // Keywords to trigger this recipe in search
    std::vector<const char*> keywords;
    
    // Structural procedure steps
    std::vector<WorkflowStep> steps;
    
    // The key IPC methods involved in this workflow
    std::vector<const char*> key_methods;
};

class WorkflowRecipeRegistry {
public:
    static WorkflowRecipeRegistry& instance();
    
    const std::vector<WorkflowRecipe>& all() const;
    std::vector<const WorkflowRecipe*> search(const std::string& query) const;
    // Exact lookup by id, for agent.get_examples {"workflow": "..."}.
    const WorkflowRecipe* find(const std::string& id) const;

private:
    WorkflowRecipeRegistry();
    std::vector<WorkflowRecipe> m_recipes;
};
