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

#include "OOCManager.h"
#include "CacheLoader.h"

#include <plugin/io/db/DBConnector.h>

#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>

#include <core/brayns/engineapi/Camera.h>
#include <core/brayns/engineapi/FrameBuffer.h>
#include <core/brayns/engineapi/Model.h>
#include <core/brayns/engineapi/Scene.h>

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
namespace io
{
using namespace common;
using namespace db;

OOCManager::OOCManager(Scene& scene, const Camera& camera, const CommandLineArguments& arguments)
    : _scene(scene)
    , _camera(camera)
{
    _parseArguments(arguments);

    PLUGIN_INFO(3, "=================================");
    PLUGIN_INFO(3, "Out-Of-Core engine is now enabled");
    PLUGIN_INFO(3, "---------------------------------");
    PLUGIN_INFO(3, "DB Connection string: " << _dbConnectionString);
    PLUGIN_INFO(3, "DB Schema           : " << _dbSchema);
    PLUGIN_INFO(3, "Description         : " << _sceneConfiguration.description);
    PLUGIN_INFO(3, "Update frequency    : " << _updateFrequency);
    PLUGIN_INFO(3, "Scene size          : " << _sceneConfiguration.sceneSize);
    PLUGIN_INFO(3, "Brick size          : " << _sceneConfiguration.brickSize);
    PLUGIN_INFO(3, "Nb of bricks        : " << _sceneConfiguration.nbBricks);
    PLUGIN_INFO(3, "Visible bricks      : " << _nbVisibleBricks);
    PLUGIN_INFO(3, "Bricks per cycle    : " << _nbBricksPerCycle);
    PLUGIN_INFO(3, "Unload bricks       : " << (_unloadBricks ? "On" : "Off"));
    PLUGIN_INFO(3, "=================================");
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

    uint32_t nbLoads = 0;
    double totalLoadingTime = 0.f;

    while (true)
    {
        const Vector3d& cameraPosition = _camera.getPosition();
        const Vector3i brick = (cameraPosition - _sceneConfiguration.brickSize / 2.0) / _sceneConfiguration.brickSize;
        const int32_t brickId = brick.z + brick.y * _sceneConfiguration.nbBricks +
                                brick.x * _sceneConfiguration.nbBricks * _sceneConfiguration.nbBricks;

        if (_frameBuffer && _frameBuffer->numAccumFrames() > 1)
        {
            bricksToLoad.clear();

            // Identify visible bricks (the ones surrounding the camera)
            std::set<int32_t> visibleBricks;
            for (int32_t x = 0; x < _nbVisibleBricks; ++x)
                for (int32_t y = 0; y < _nbVisibleBricks; ++y)
                    for (int32_t z = 0; z < _nbVisibleBricks; ++z)
                    {
                        visibleBricks.insert((z + brick.z) + (y + brick.y) * _sceneConfiguration.nbBricks +
                                             (x + brick.x) * _sceneConfiguration.nbBricks *
                                                 _sceneConfiguration.nbBricks);
                        visibleBricks.insert((-z + brick.z) + (-y + brick.y) * _sceneConfiguration.nbBricks +
                                             (-x + brick.x) * _sceneConfiguration.nbBricks *
                                                 _sceneConfiguration.nbBricks);
                    }

            // Identify bricks to load
            for (const int32_t visibleBrick : visibleBricks)
                if (std::find(loadedBricks.begin(), loadedBricks.end(), visibleBrick) == loadedBricks.end())
                {
                    bricksToLoad.insert(visibleBrick);
                    if (bricksToLoad.size() >= _nbBricksPerCycle)
                        break;
                }

            if (!bricksToLoad.empty())
                PLUGIN_INFO(3, "Loading bricks   " << int32_set_to_string(bricksToLoad));

            _progress = double(bricksToLoad.size()) / double(_nbBricksPerCycle);
            // Loading bricks
            if (!bricksToLoad.empty())
            {
                const auto brickToLoad = (*bricksToLoad.begin());
                loadedBricks.insert(brickToLoad);
                const auto start = std::chrono::steady_clock::now();
                try
                {
                    modelsToAddToScene = loader.importBrickFromDB(brickToLoad);
                }
                catch (std::runtime_error& e)
                {
                    PLUGIN_DEBUG(e.what());
                }
                const auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

                totalLoadingTime += duration.count();
                ++nbLoads;

                bricksToLoad.erase(bricksToLoad.begin());
            }

            if (bricksToLoad.size())
            {
                PLUGIN_DEBUG("Current brick Id: " << brickId);
                PLUGIN_DEBUG("Loaded bricks   : " << int32_set_to_string(loadedBricks));
                PLUGIN_DEBUG("Visible bricks  : " << int32_set_to_string(visibleBricks));

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
                            std::find(visibleBricks.begin(), visibleBricks.end(), id) != visibleBricks.end();
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
                        const auto it = std::find(visibleBricks.begin(), visibleBricks.end(), loadedBrick);
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
                    PLUGIN_DEBUG("Removing model: " << md->getModelID());
                    _scene.removeModel(md->getModelID());
                    sceneModified = true;
                }
                modelsToRemoveFromScene.clear();
            }

            for (auto md : modelsToAddToScene)
            {
                md->setVisible(true);
                _scene.addModel(md);
                PLUGIN_DEBUG("Adding model: " << md->getModelID());
                sleep(_updateFrequency);
                sceneModified = true;
            }
            modelsToAddToScene.clear();

            for (auto md : modelsToShow)
            {
                PLUGIN_DEBUG("Making model visible: " << md->getModelID());
                md->setVisible(true);
                sceneModified = true;
            }
            modelsToShow.clear();
            if (sceneModified)
                _scene.markModified(false);
        }

        sleep(_updateFrequency);
        _averageLoadingTime = totalLoadingTime / nbLoads;
        PLUGIN_DEBUG("Average loading time (ms): " << _averageLoadingTime);
        previousBrickId = brickId;
    }
}

void OOCManager::_parseArguments(const CommandLineArguments& arguments)
{
    std::string dbHost, dbPort, dbUser, dbPassword, dbName;
    for (const auto& argument : arguments)
    {
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

    // Configuration
    auto& connector = DBConnector::getInstance();
    _sceneConfiguration = connector.getSceneConfiguration();

    const bool disableBroadcasting = std::getenv(ENV_ROCKETS_DISABLE_SCENE_BROADCASTING.c_str()) != nullptr;
    if (!disableBroadcasting)
        PLUGIN_THROW(ENV_ROCKETS_DISABLE_SCENE_BROADCASTING +
                     " environment variable must be set when out-of-core is enabled");
}
} // namespace io
} // namespace bioexplorer
