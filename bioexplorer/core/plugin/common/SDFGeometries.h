/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "Node.h"
#include "ThreadSafeContainer.h"
#include "Types.h"

namespace bioexplorer
{
namespace common
{
using namespace common;
using namespace brayns;

/**
 * @brief The SDFGeometries abstract class
 */
class SDFGeometries : public common::Node
{
public:
    /**
     * @brief Construct a new SDFGeometries object
     *
     */
    SDFGeometries(const double radiusMultiplier,
                  const Vector3d& scale = Vector3d(1.0, 1.0, 1.0));

    void addSDFDemo(Model& model);

protected:
    double _radiusMultiplier{1.0};
};

} // namespace common
} // namespace bioexplorer