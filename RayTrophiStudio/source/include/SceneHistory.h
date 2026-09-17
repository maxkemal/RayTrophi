/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SceneHistory.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "SceneCommand.h"
#include <deque>
#include <memory>

// ============================================================================
// SCENE HISTORY - Undo/Redo Stack Manager
// ============================================================================
// Manages command history with configurable max depth.
// Automatically clears redo stack when new command is executed.
// Thread-safe for single-threaded UI context.
// ============================================================================

class SceneHistory {
public:
    SceneHistory(size_t max_history = 50) : max_history_(max_history) {}
    
    // Record a new command (clears redo stack)
    void record(std::unique_ptr<SceneCommand> command);
    
    // Undo last command
    // outHandledUiCacheSync (optional) receives the applied command's
    // handlesUiCacheSync(): true means the command already synced the SceneUI
    // caches and the caller must NOT invalidate them.
    bool undo(UIContext& ctx, bool* outHandledUiCacheSync = nullptr);
    
    // Redo last undone command
    bool redo(UIContext& ctx, bool* outHandledUiCacheSync = nullptr);
    
    // Check if undo/redo available
    bool canUndo() const { return !undo_stack_.empty(); }
    bool canRedo() const { return !redo_stack_.empty(); }
    
    // Get description of next undo/redo
    std::string getUndoDescription() const;
    std::string getRedoDescription() const;
    
    // Clear all history
    void clear();
    
    // Get history size
    size_t getUndoCount() const { return undo_stack_.size(); }
    size_t getRedoCount() const { return redo_stack_.size(); }
    
private:
    std::deque<std::unique_ptr<SceneCommand>> undo_stack_;
    std::deque<std::unique_ptr<SceneCommand>> redo_stack_;
    size_t max_history_;
};

