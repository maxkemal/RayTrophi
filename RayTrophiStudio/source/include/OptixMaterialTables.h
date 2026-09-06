/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          OptixMaterialTables.h
* =========================================================================
*
* THE GLOBAL MATERIAL / TEXTURE / VOLUMETRIC TABLES AN OptiX LAUNCH NEEDS.
*
* ★ This lived as AssimpLoader::convertTrianglesToOptixData and contained no
* Assimp whatsoever: it reads MaterialManager and produces GPU-side tables. It
* only lived there because AssimpLoader was, historically, where "everything
* about materials" ended up. It was one of the last two things keeping a live
* AssimpLoader instance inside Renderer, so it had to move before the loader
* could be deleted (CLAUDE.md rule 5 — renamed, because the old name described
* a triangle conversion it does not really do).
*
* ★★ THE TRIANGLE LIST IS NOT THE POINT, AND THE NAME USED TO LIE.
* Geometry extraction inside is gated on a non-empty list, but the tables are
* pulled from MaterialManager REGARDLESS. A flat-SoA scene passes an EMPTY
* triangle vector and still needs this call — without it the material buffer is
* null and optixLaunch dies with an illegal memory access (CUDA 700). Do not
* "optimise" the empty-list case away.
* =========================================================================
*/
#pragma once

#include <memory>
#include <vector>

#include "OptixTypes.h"

class Triangle;

// Builds the canonical material/texture/volumetric tables from MaterialManager.
// Safe to call with an empty triangle list; see the header note above.
OptixGeometryData buildOptixMaterialTables(const std::vector<std::shared_ptr<Triangle>>& triangles);
