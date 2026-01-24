/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          PyPrimitives.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYPRIMITIVES_HAS_BEEN_INCLUDED
#define NANOVDB_PYPRIMITIVES_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> void definePrimitives(nb::module_& m);

} // namespace pynanovdb

#endif

