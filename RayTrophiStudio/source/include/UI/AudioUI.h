#pragma once
struct UIContext;
struct AssetRecord;
namespace AudioUI {
void addSource(UIContext& ctx);
void appendAsset(UIContext& ctx, const AssetRecord& asset, float x, float y, float z);
bool editSelection(UIContext& ctx, bool duplicate);
void drawAssetActions(UIContext& ctx, const AssetRecord& asset);
void drawHierarchy(UIContext& ctx);
void drawProperties(UIContext& ctx);
void drawViewport(UIContext& ctx, bool orthographic, bool showGizmos, bool& hit);
void updateListener(UIContext& ctx);
}
