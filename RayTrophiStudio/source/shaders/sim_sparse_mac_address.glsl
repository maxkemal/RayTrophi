#ifndef RT_SPARSE_MAC_ADDRESS
#define RT_SPARSE_MAC_ADDRESS

// Map stores slot + 1; zero is background. The positive-side tile owns an
// interior MAC face. The final domain face belongs to the last cell's tile.
uvec3 sparseMacTileDims(ivec3 cells) {
    return (uvec3(cells) + 7u) / 8u;
}
uint sparseMacTileKey(ivec3 face, int component, ivec3 cells) {
    ivec3 owner = face;
    owner[component] = min(owner[component], cells[component] - 1);
    uvec3 tile = uvec3(owner) / 8u;
    uvec3 dims = sparseMacTileDims(cells);
    return tile.x + dims.x * (tile.y + dims.y * tile.z);
}
uint sparseMacLocal(ivec3 face, int component, ivec3 cells) {
    ivec3 owner = face;
    owner[component] = min(owner[component], cells[component] - 1);
    uvec3 local = uvec3(face - (owner / 8) * 8);
    uvec3 dims = uvec3(8u);
    dims[component] = 9u;
    return local.x + dims.x * (local.y + dims.y * local.z);
}
ivec3 sparseMacFace(uint lane, int component, uint key, ivec3 cells) {
    uvec3 tiles = sparseMacTileDims(cells);
    uvec3 tile = uvec3(key % tiles.x, (key / tiles.x) % tiles.y,
                       key / (tiles.x * tiles.y));
    uvec3 dims = uvec3(8u);
    dims[component] = 9u;
    uint local = lane % 576u;
    uvec3 face = tile * 8u + uvec3(local % dims.x,
        (local / dims.x) % dims.y, local / (dims.x * dims.y));
    return ivec3(face);
}
bool sparseMacOwned(ivec3 face, int component, uint key, ivec3 cells) {
    ivec3 maximum = cells - ivec3(1);
    maximum[component] += 1;
    return all(greaterThanEqual(face, ivec3(0))) &&
        all(lessThanEqual(face, maximum)) &&
        sparseMacTileKey(face, component, cells) == key;
}
uint sparseMacDenseIndex(ivec3 face, int component, ivec3 cells) {
    uvec3 dims = uvec3(cells);
    dims[component] += 1u;
    return uint(face.x) + dims.x * (uint(face.y) + dims.y * uint(face.z));
}
#endif
