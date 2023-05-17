/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <brayns/common/simulation/AbstractSimulationHandler.h>

#include <brayns/common/types.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace fields
{
using namespace brayns;

/**
 * @brief The FieldsHandler class handles electro-magnetic fields data
 * structures
 */
class FieldsHandler : public AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    FieldsHandler(const Scene& scene, const double voxelSize,
                  const double density);

    /**
     * @brief Construct a new FieldsHandler from a file
     *
     * @param filename Full path of the file
     */
    FieldsHandler(const std::string& filename);

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
     * @brief Clone the AbstractSimulationHandler
     *
     * @return AbstractSimulationHandlerPtr Clone of the
     * AbstractSimulationHandler
     */
    AbstractSimulationHandlerPtr clone() const final;

    /**
     * @brief Export the octree information to a file
     *
     * @param filename Full path of the file
     */
    void exportToFile(const std::string& filename) const;

    /**
     * @brief Import the octree information from a file
     *
     * @param filename Full path of the file
     */
    void importFromFile(const std::string& filename);

    /**
     * @brief Get the Dimensions of the octree
     *
     * @return const Vector3ui& Dimensions of the octree
     */
    const Vector3ui& getDimensions() const { return _dimensions; }

    /**
     * @brief Get the voxel spacing information
     *
     * @return const Vector3f& The voxel spacing information
     */
    const Vector3f& getSpacing() const { return _spacing; }

    /**
     * @brief Get the offset of the octree
     *
     * @return const Vector3f& Offset of the octree
     */
    const Vector3f& getOffset() const { return _offset; }

private:
    void _buildOctree(const Scene& scene, const double voxelSize,
                      const double density);

    Vector3ui _dimensions;
    Vector3f _spacing;
    Vector3f _offset;
};
typedef std::shared_ptr<FieldsHandler> FieldsHandlerPtr;
} // namespace fields
} // namespace bioexplorer
