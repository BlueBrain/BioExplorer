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

#include "Types.h"

#include <plugin/common/Logs.h>

namespace bioexplorer
{
/**
 * @brief GeneralSettings is a singleton class that holds general settings for
 * the plugin
 *
 */
class GeneralSettings
{
public:
    static GeneralSettings* getInstance()
    {
        if (!_instance)
            _instance = new GeneralSettings();
        PLUGIN_WARN << "_instance=" << _instance << std::endl;
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
    void setModelVisibilityOnCreation(const bool value)
    {
        _modelVisibilityOnCreation = value;
    }

    /**
     * @brief Get the Off Folder object. The off folder is the cache where Off
     * files are stored when using the Surface representation of molecules
     *
     * @return The path to the folder
     */
    std::string getOffFolder() { return _offFolder; }

    /**
     * @brief Set the Off folder location
     *
     * @param value Path to the folder
     */
    void setOffFolder(const std::string& value) { _offFolder = value; }

    std::string getBricksFolder() { return _bricksFolder; }
    void setBricksFolder(const std::string& value) { _bricksFolder = value; }

    std::string getDatabaseConnectionString() { return _dbConnectionString; }
    void setDatabaseConnectionString(const std::string& value)
    {
        _dbConnectionString = value;
    }

    std::string getDatabaseSchema() { return _dbSchema; }
    void setDatabaseSchema(const std::string& value) { _dbSchema = value; }

private:
    GeneralSettings() {}
    ~GeneralSettings() {}

    bool _modelVisibilityOnCreation{true};
    std::string _offFolder{"/tmp/"};
    std::string _bricksFolder{"/tmp/"};
    std::string _dbConnectionString;
    std::string _dbSchema;

    static GeneralSettings* _instance;
};
} // namespace bioexplorer
