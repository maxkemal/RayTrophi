#include "SceneHistory.h"
#include "globals.h"  // For SCENE_LOG_INFO


void SceneHistory::record(std::unique_ptr<SceneCommand> command) {
    // Clear redo stack (new action invalidates redo)
    redo_stack_.clear();
    
    // Add to undo stack
    undo_stack_.push_back(std::move(command));
    
    // Limit Configuration
    const size_t LIMIT_HEAVY = 5;      // Delete, Duplicate
    const size_t LIMIT_TRANSFORM = 20; // Move, Rotate, Scale
    const size_t LIMIT_GLOBAL = 50;    // Safety cap
    
    // 1. Enforce Type Specific Limits
    {
        size_t count_heavy = 0;
        size_t count_transform = 0;
        
        // Count existing commands
        for (const auto& cmd : undo_stack_) {
            if (cmd->getType() == SceneCommand::Type::Heavy) count_heavy++;
            else if (cmd->getType() == SceneCommand::Type::Transform) count_transform++;
        }
        
        // Prune if needed (Linear scan and remove oldest matches)
        if (count_heavy > LIMIT_HEAVY) {
            for (auto it = undo_stack_.begin(); it != undo_stack_.end(); ) {
                if ((*it)->getType() == SceneCommand::Type::Heavy) {
                    it = undo_stack_.erase(it); // Remove OLDER heavy command
                    count_heavy--;
                    if (count_heavy <= LIMIT_HEAVY) break;
                } else {
                    ++it;
                }
            }
        }
        
        if (count_transform > LIMIT_TRANSFORM) {
            for (auto it = undo_stack_.begin(); it != undo_stack_.end(); ) {
                if ((*it)->getType() == SceneCommand::Type::Transform) {
                    it = undo_stack_.erase(it); // Remove OLDER transform command
                    count_transform--;
                    if (count_transform <= LIMIT_TRANSFORM) break;
                } else {
                    ++it;
                }
            }
        }
    }

    // 2. Enforce Global Limit (FIFO)
    while (undo_stack_.size() > LIMIT_GLOBAL) {
        undo_stack_.pop_front();
    }
    
    // SCENE_LOG_INFO("History: Recorded command (undo: " + 
    //                std::to_string(undo_stack_.size()) + ")");  // Too verbose
}

bool SceneHistory::undo(UIContext& ctx) {
    if (undo_stack_.empty()) {
        return false;
    }
    
    // Pop command from undo stack
    auto command = std::move(undo_stack_.back());
    undo_stack_.pop_back();
    
    // Execute undo
    SCENE_LOG_INFO("History: Undo - " + command->getDescription());
    command->undo(ctx);
    
    // Push to redo stack
    redo_stack_.push_back(std::move(command));
    
    return true;
}

bool SceneHistory::redo(UIContext& ctx) {
    if (redo_stack_.empty()) {
        return false;
    }
    
    // Pop command from redo stack
    auto command = std::move(redo_stack_.back());
    redo_stack_.pop_back();
    
    // Execute command again
    SCENE_LOG_INFO("History: Redo - " + command->getDescription());
    command->execute(ctx);
    
    // Push back to undo stack
    undo_stack_.push_back(std::move(command));
    
    return true;
}

std::string SceneHistory::getUndoDescription() const {
    if (undo_stack_.empty()) {
        return "";
    }
    return undo_stack_.back()->getDescription();
}

std::string SceneHistory::getRedoDescription() const {
    if (redo_stack_.empty()) {
        return "";
    }
    return redo_stack_.back()->getDescription();
}

void SceneHistory::clear() {
    undo_stack_.clear();
    redo_stack_.clear();
    SCENE_LOG_INFO("History: Cleared");
}
