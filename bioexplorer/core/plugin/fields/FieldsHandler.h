/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <brayns/api.h>
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
class FieldsHandler : public brayns::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    FieldsHandler(const Scene& scene, const float voxelSize,
                  const float density);

    /**
     * @brief Construct a new Fields Handler object
     *
     * @param filename
     */
    FieldsHandler(const std::string& filename);

    /**
     * @brief Construct a new Fields Handler object
     *
     * @param rhs
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
     * @return void*
     */
    void* getFrameData(const uint32_t) final;

    /**
     * @brief
     *
     * @return true
     * @return false
     */
    bool isReady() const final { return true; }

    /**
     * @brief
     *
     * @return brayns::AbstractSimulationHandlerPtr
     */
    brayns::AbstractSimulationHandlerPtr clone() const final;

    /**
     * @brief
     *
     * @param filename
     */
    void exportToFile(const std::string& filename);

    /**
     * @brief
     *
     * @param filename
     */
    void importFromFile(const std::string& filename);

    /**
     * @brief Get the Dimensions object
     *
     * @return const glm::uvec3&
     */
    const glm::uvec3& getDimensions() const { return _dimensions; }

    /**
     * @brief Get the Spacing object
     *
     * @return const glm::vec3&
     */
    const glm::vec3& getSpacing() const { return _spacing; }

    /**
     * @brief Get the Offset object
     *
     * @return const glm::vec3&
     */
    const glm::vec3& getOffset() const { return _offset; }

private:
    void _buildOctree(const Scene& scene, const float voxelSize,
                      const float density);

    glm::uvec3 _dimensions;
    glm::vec3 _spacing;
    glm::vec3 _offset;
};
typedef std::shared_ptr<FieldsHandler> FieldsHandlerPtr;
} // namespace fields
} // namespace bioexplorer
