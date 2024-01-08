/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include <platform/core/common/simulation/AbstractSimulationHandler.h>

#include <platform/core/common/Types.h>

namespace bioexplorer
{
namespace fields
{
/**
 * @brief The FieldsHandler class handles electro-magnetic fields data
 * structures
 */
class FieldsHandler : public core::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    FieldsHandler(core::Engine& engine, core::Model& model, const double voxelSize, const double density,
                  const uint32_ts& modelIds);

    /**
     * @brief Construct a new FieldsHandler object
     *
     * @param rhs A copy of the FieldsHandler object
     */
    FieldsHandler(const FieldsHandler& rhs);

    /**
     * @brief Destroy the Fields Handler object
     *
     */
    ~FieldsHandler();

    /**
     * @brief Get the Frame Data object
     *
     * @return void* A buffer of the data for the current frame
     */
    void* getFrameData(const uint32_t) final;

    /**
     * @brief Current state of the handler
     *
     * @return true The data is loaded in memory and available
     * @return false The data is not yet available
     */
    bool isReady() const final { return true; }

    /**
     * @brief Get the Dimensions of the octree
     *
     * @return const Vector3ui& Dimensions of the octree
     */
    const core::Vector3ui& getDimensions() const { return _dimensions; }

    /**
     * @brief Get the voxel spacing information
     *
     * @return const Vector3f& The voxel spacing information
     */
    const core::Vector3f& getSpacing() const { return _spacing; }

    /**
     * @brief Get the offset of the octree
     *
     * @return const Vector3f& Offset of the octree
     */
    const core::Vector3f& getOffset() const { return _offset; }

    ::core::Model* getModel() { return _model; }

protected:
    virtual void _buildOctree() = 0;

    core::Vector3ui _dimensions;
    core::Vector3f _spacing;
    core::Vector3f _offset;

    core::Engine& _engine;
    uint32_ts _modelIds;
    core::Model* _model{nullptr};
    double _voxelSize;
    double _density;

    bool _octreeInitialized{false};
};
typedef std::shared_ptr<FieldsHandler> FieldsHandlerPtr;
} // namespace fields
} // namespace bioexplorer
