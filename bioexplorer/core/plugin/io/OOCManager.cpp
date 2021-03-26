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

#include "OOCManager.h"
#include "CacheLoader.h"

#include <plugin/io/db/DBConnector.h>

#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>

#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/FrameBuffer.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

#include <thread>

namespace
{
std::string int32_set_to_string(const std::set<int32_t>& s)
{
    std::string str = "[";
    uint32_t count = 0;
    for (const auto v : s)
    {
        if (count > 0)
            str += ", ";
        str += std::to_string(v);
        ++count;
    }
    str += "]";
    return str;
}
} // namespace

namespace bioexplorer
{
OOCManager::OOCManager(Scene& scene, const Camera& camera,
                       const CommandLineArguments& arguments)
    : _scene(scene)
    , _camera(camera)
{
    _parseArguments(arguments);

    PLUGIN_INFO << "=================================" << std::endl;
    PLUGIN_INFO << "Out-Of-Core engine is now enabled" << std::endl;
    PLUGIN_INFO << "---------------------------------" << std::endl;
    PLUGIN_INFO << "DB Connection string: " << _dbConnectionString << std::endl;
    PLUGIN_INFO << "DB Schema           : " << _dbSchema << std::endl;
    PLUGIN_INFO << "Description         : " << _description << std::endl;
    PLUGIN_INFO << "Update frequency    : " << _updateFrequency << std::endl;
    PLUGIN_INFO << "Scene size          : " << _sceneSize << std::endl;
    PLUGIN_INFO << "Brick size          : " << _brickSize << std::endl;
    PLUGIN_INFO << "Nb of bricks        : " << _nbBricks << std::endl;
    PLUGIN_INFO << "Visible bricks      : " << _nbVisibleBricks << std::endl;
    PLUGIN_INFO << "Bricks per cycle    : " << _nbBricksPerCycle << std::endl;
    PLUGIN_INFO << "Unload bricks       : " << (_unloadBricks ? "On" : "Off")
                << std::endl;
    PLUGIN_INFO << "=================================" << std::endl;
}

void OOCManager::loadBricks()
{
    GeneralSettings::getInstance()->setModelVisibilityOnCreation(false);
    std::thread t(&OOCManager::_loadBricks, this);
    t.detach();
}

void OOCManager::_loadBricks()
{
    std::set<int32_t> loadedBricks;
    std::set<int32_t> bricksToLoad;
    std::vector<ModelDescriptorPtr> modelsToAddToScene;
    std::vector<ModelDescriptorPtr> modelsToRemoveFromScene;
    std::vector<ModelDescriptorPtr> modelsToShow;
    int32_t previousBrickId{std::numeric_limits<int32_t>::max()};
    CacheLoader loader(_scene);
    DBConnector connector(_dbConnectionString, _dbSchema);

    while (true)
    {
        const Vector3f& cameraPosition = _camera.getPosition();
        const Vector3i brick = cameraPosition / _brickSize;
        const int32_t brickId =
            brick.z + brick.y * _nbBricks + brick.x * _nbBricks * _nbBricks;

        if (_frameBuffer && _frameBuffer->getAccumFrames() > 1)
        {
            bricksToLoad.clear();

            // Identify visible bricks (the ones surrounding the camera)
            std::set<int32_t> visibleBricks;
            for (int32_t x = -_nbVisibleBricks; x < _nbVisibleBricks; ++x)
                for (int32_t y = -_nbVisibleBricks; y < _nbVisibleBricks; ++y)
                    for (int32_t z = -_nbVisibleBricks; z < _nbVisibleBricks;
                         ++z)
                        visibleBricks.insert(
                            (z + brick.z) + (y + brick.y) * _nbBricks +
                            (x + brick.x) * _nbBricks * _nbBricks);

            // Identify bricks to load
            for (const int32_t visibleBrick : visibleBricks)
                if (std::find(loadedBricks.begin(), loadedBricks.end(),
                              visibleBrick) == loadedBricks.end())
                {
                    bricksToLoad.insert(visibleBrick);
                    if (bricksToLoad.size() >= _nbBricksPerCycle)
                        break;
                }

            if (!bricksToLoad.empty())
                PLUGIN_INFO << "Loading bricks   "
                            << int32_set_to_string(bricksToLoad) << std::endl;

            // Loading bricks
            if (!bricksToLoad.empty())
            {
                const auto brickToLoad = (*bricksToLoad.begin());
                loadedBricks.insert(brickToLoad);
                try
                {
#ifdef USE_PQXX
                    modelsToAddToScene =
                        loader.importBrickFromDB(connector, brickToLoad);
#else
                    char idStr[7];
                    sprintf(idStr, "%06d", brickToLoad);
                    const std::string filename =
                        _bricksFolder + "/brick" + idStr + ".bioexplorer";
                    modelsToAddToScene =
                        loader.importModelsFromFile(filename, brickToLoad);
#endif
                }
                catch (std::runtime_error& e)
                {
                    PLUGIN_DEBUG << e.what() << std::endl;
                }
                bricksToLoad.erase(bricksToLoad.begin());
            }

            if (bricksToLoad.size())
            {
                PLUGIN_DEBUG << "Current brick Id: " << brickId << std::endl;
                PLUGIN_DEBUG
                    << "Loaded bricks   : " << int32_set_to_string(loadedBricks)
                    << std::endl;
                PLUGIN_DEBUG << "Visible bricks  : "
                             << int32_set_to_string(visibleBricks) << std::endl;

                bool visibilityModified = false;

                // Make visible models visible and remove invisible models
                auto modelDescriptors = _scene.getModelDescriptors();
                for (auto modelDescriptor : modelDescriptors)
                {
                    const auto metadata = modelDescriptor->getMetadata();
                    const auto it = metadata.find(METADATA_BRICK_ID);
                    if (it != metadata.end())
                    {
                        const int32_t id = atoi(it->second.c_str());
                        const bool visible =
                            std::find(visibleBricks.begin(),
                                      visibleBricks.end(),
                                      id) != visibleBricks.end();
                        if (visible)
                        {
                            if (!modelDescriptor->getVisible())
                                modelsToShow.push_back(modelDescriptor);
                        }
                        else
                            modelsToRemoveFromScene.push_back(modelDescriptor);
                    }
                }

                if (_unloadBricks)
                {
                    // Prevent invisible bricks from being loaded
                    auto i = loadedBricks.begin();
                    while (i != loadedBricks.end())
                    {
                        const auto loadedBrick = (*i);
                        const auto it =
                            std::find(visibleBricks.begin(),
                                      visibleBricks.end(), loadedBrick);
                        if (it == visibleBricks.end())
                            loadedBricks.erase(i++);
                        else
                            ++i;
                    }
                }
            }

            bool sceneModified = false;
            if (_unloadBricks)
            {
                for (auto md : modelsToRemoveFromScene)
                {
                    PLUGIN_DEBUG << "Removing model: " << md->getModelID()
                                 << std::endl;
                    _scene.removeModel(md->getModelID());
                    sceneModified = true;
                }
                modelsToRemoveFromScene.clear();
            }

            for (auto md : modelsToAddToScene)
            {
                md->setVisible(true);
                _scene.addModel(md);
                PLUGIN_DEBUG << "Adding model: " << md->getModelID()
                             << std::endl;
                sleep(_updateFrequency);
                sceneModified = true;
            }
            modelsToAddToScene.clear();

            for (auto md : modelsToShow)
            {
                PLUGIN_DEBUG << "Making model visible: " << md->getModelID()
                             << std::endl;
                md->setVisible(true);
                sceneModified = true;
            }
            modelsToShow.clear();
            if (sceneModified)
                _scene.markModified(false);
        }

        sleep(_updateFrequency);
        previousBrickId = brickId;
    }
}

void OOCManager::_parseArguments(const CommandLineArguments& arguments)
{
    std::string dbHost, dbPort, dbUser, dbPassword, dbName;
    for (const auto& argument : arguments)
    {
#ifdef USE_PQXX
        if (argument.first == ARG_OOC_DB_HOST)
            dbHost = argument.second;
        if (argument.first == ARG_OOC_DB_PORT)
            dbPort = argument.second;
        if (argument.first == ARG_OOC_DB_USER)
            dbUser = argument.second;
        if (argument.first == ARG_OOC_DB_PASSWORD)
            dbPassword = argument.second;
        if (argument.first == ARG_OOC_DB_NAME)
            dbName = argument.second;
        if (argument.first == ARG_OOC_DB_SCHEMA)
            _dbSchema = argument.second;
#else
        if (argument.first == ARG_OOC_BRICKS_FOLDER)
            _bricksFolder = argument.second;
#endif
        if (argument.first == ARG_OOC_UPDATE_FREQUENCY)
            _updateFrequency = atof(argument.second.c_str());
        if (argument.first == ARG_OOC_VISIBLE_BRICKS)
            _nbVisibleBricks = atoi(argument.second.c_str());
        if (argument.first == ARG_OOC_UNLOAD_BRICKS)
            _unloadBricks = true;
        if (argument.first == ARG_OOC_SHOW_GRID)
            _showGrid = true;
        if (argument.first == ARG_OOC_NB_BRICKS_PER_CYCLE)
            _nbBricksPerCycle = atoi(argument.second.c_str());
    }

    // Sanity checks
    _dbConnectionString = "host=" + dbHost + " port=" + dbPort +
                          " dbname=" + dbName + " user=" + dbUser +
                          " password=" + dbPassword;

    // Configuration
    DBConnector connector(_dbConnectionString, _dbSchema);
    connector.getConfiguration(_description, _sceneSize, _nbBricks);
    if (_nbBricks == 0)
        PLUGIN_THROW(std::runtime_error("Invalid number of bricks)"));
    _brickSize = _sceneSize / _nbBricks;

    const bool disableBroadcasting =
        std::getenv(ENV_ROCKETS_DISABLE_SCENE_BROADCASTING.c_str()) != nullptr;
    if (!disableBroadcasting)
        PLUGIN_THROW(std::runtime_error(
            ENV_ROCKETS_DISABLE_SCENE_BROADCASTING +
            " environment variable must be set when out-of-core is enabled"));
}

} // namespace bioexplorer
