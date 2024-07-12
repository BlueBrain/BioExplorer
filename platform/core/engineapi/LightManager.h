/*
    Copyright 2019 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <platform/core/common/Api.h>
#include <platform/core/common/BaseObject.h>
#include <platform/core/common/Types.h>

#include <map>
#include <unordered_map>

namespace core
{
/**
 * @class LightManager
 * @extends BaseObject
 * @brief Manages light sources in a scene
 *
 * A LightManager object is responsible for managing light sources in a scene.
 * It provides methods to add, remove, retrieve, and clear light sources.
 */
class LightManager : public BaseObject
{
public:
    /**
     * @brief addLight
     * Attaches a light source to the scene.
     * @param light Pointer to an object representing the light source.
     * @return The ID assigned to the new light source.
     */
    PLATFORM_API size_t addLight(LightPtr light);

    /**
     * @brief removeLight
     * Removes a light source from the scene for a given ID.
     * @param id ID of the light source to be removed.
     */
    PLATFORM_API void removeLight(const size_t id);

    /**
     * @brief removeLight
     * Removes a light source from the scene.
     * @param light Pointer to the light source to be removed.
     */
    PLATFORM_API void removeLight(LightPtr light);

    /**
     * @brief getLight
     * Gets a light source from the scene for a given ID.
     * Note: If changing the light then call markModified to propagate the
     * changes.
     * @param id ID of the light to retrieve.
     * @return Pointer to the requested light source, or nullptr if not found.
     */
    PLATFORM_API LightPtr getLight(const size_t id);

    /**
     * @brief getLights
     * Gets all light sources currently managed by the LightManager object.
     * @return Immutable list of all light sources and their IDs.
     */
    PLATFORM_API const std::map<size_t, LightPtr>& getLights() const;

    /**
     * @brief clearLights
     * Removes all light sources managed by the LightManager object.
     */
    PLATFORM_API void clearLights();

private:
    std::map<size_t, LightPtr> _lights;                  // Map containing all light sources.
    std::unordered_map<LightPtr, size_t> _lightsInverse; // Inverse mapping of light sources to their IDs.
    size_t _IDctr{0};                                    // Internal counter for assigning IDs to new light sources.
};
} // namespace core
