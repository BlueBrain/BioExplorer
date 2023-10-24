/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include "FieldsHandler.h"

#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Scene.h>

namespace bioexplorer
{
namespace fields
{
/**
 * @brief The VectorFieldsHandler class handles electro-magnetic fields data
 * structures
 */
class VectorFieldsHandler : public FieldsHandler
{
public:
    /**
     * @brief Default constructor
     */
    VectorFieldsHandler(const core::Scene& scene, core::Model& model, const double voxelSize, const double density,
                        const uint32_ts& modelIds);

    /**
     * @brief Construct a new VectorFieldsHandler from a file
     *
     * @param filename Full path of the file
     */
    VectorFieldsHandler(core::Scene& scene, const std::string& filename);

    /**
     * @brief Clone the AbstractSimulationHandler
     *
     * @return AbstractSimulationHandlerPtr Clone of the  AbstractSimulationHandler
     */
    core::AbstractSimulationHandlerPtr clone() const final;

private:
    void _buildOctree() final;
};
typedef std::shared_ptr<VectorFieldsHandler> VectorFieldsHandlerPtr;
} // namespace fields
} // namespace bioexplorer
