/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "Types.h"

namespace bioexplorer
{
namespace common
{
/**
 * @brief GeneralSettings is a singleton class that holds general settings for
 * the plugin
 *
 */
class GeneralSettings
{
public:
    /**
     * @brief Construct a new General Settings object
     *
     */
    GeneralSettings() {}

    /**
     * @brief Get the Instance object
     *
     * @return GeneralSettings* Pointer to the object
     */
    static GeneralSettings* getInstance()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_instance)
            _instance = new GeneralSettings();
        return _instance;
    }

    /**
     * @brief Get the Model Visibility On Creation object. This is used to
     * optimize the loading of objects in the scene. By default, every addition
     * will require the rendering engine to update the BVH, and that takes time.
     * By making the model invisible, it is created but not added to the scene.
     * The set_models_visibility function can be used to make all models visible
     * at once
     */
    bool getModelVisibilityOnCreation() { return _modelVisibilityOnCreation; }

    /**
     * @brief Set the Model Visibility On Creation object. If set to false,
     * models are created but not added to the scene until they become visible
     *
     * @param value Visibility of the model on creation
     */
    void setModelVisibilityOnCreation(const bool value) { _modelVisibilityOnCreation = value; }

    /**
     * @brief Get the Off Folder object. The off folder is the cache where Off
     * files are stored when using the Surface representation of molecules
     *
     * @return The path to the folder
     */
    std::string getMeshFolder() { return _meshFolder + "/"; }

    /**
     * @brief Set the Off folder location
     *
     * @param value Path to the folder
     */
    void setMeshFolder(const std::string& value) { _meshFolder = value; }

    /**
     * @brief Get the Logging level
     *
     * @return Logging level
     */
    size_t getLoggingLevel() const { return _loggingLevel; }

    /**
     * @brief Set the Logging level
     *
     * @param value Logging level
     */
    void setLoggingLevel(const size_t value) { _loggingLevel = value; }

    /**
     * @brief Get the database logging level
     *
     * @return Database logging level
     */
    size_t getDBLoggingLevel() const { return _dbLoggingLevel; }

    /**
     * @brief Set the DB logging level
     *
     * @param value Database logging level
     */
    void setDBLoggingLevel(const size_t value) { _dbLoggingLevel = value; }

    /**
     * @brief Get the V1 Compatibility state
     *
     * @return true V1 compatibility is enabled
     * @return false V1 compatibility is disabled
     */
    bool getV1Compatibility() const { return _v1Compatibility; }

    /**
     * @brief Set the V1 compatibility state
     *
     * @param value Enabled is true, disabled otherwise
     */
    void setV1Compatibility(const bool value) { _v1Compatibility = value; }

    static std::mutex _mutex;
    static GeneralSettings* _instance;

private:
    ~GeneralSettings() {}

    bool _modelVisibilityOnCreation{true};
    std::string _meshFolder{"/tmp"};
    size_t _loggingLevel{1};
    size_t _dbLoggingLevel{0};
    bool _v1Compatibility{false};
};
} // namespace common
} // namespace bioexplorer
