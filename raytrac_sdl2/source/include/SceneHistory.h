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
    bool undo(UIContext& ctx);
    
    // Redo last undone command
    bool redo(UIContext& ctx);
    
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
