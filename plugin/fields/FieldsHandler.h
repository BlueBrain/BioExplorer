/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <brayns/common/simulation/AbstractSimulationHandler.h>

#include <brayns/api.h>
#include <brayns/common/types.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
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
    FieldsHandler(const brayns::Scene& scene, const float voxelSize);

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
    void _buildOctree(const brayns::Scene& scene, const float voxelSize);

    glm::uvec3 _dimensions;
    glm::vec3 _spacing;
    glm::vec3 _offset;
};
typedef std::shared_ptr<FieldsHandler> FieldsHandlerPtr;
} // namespace bioexplorer
