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
