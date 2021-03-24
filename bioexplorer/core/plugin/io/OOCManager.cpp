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

#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>

#include <brayns/engineapi/Camera.h>
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
OOCManager::OOCManager(Scene& scene, Camera& camera,
                       const OutOfCoreDescriptor& descriptor)
    : _scene(scene)
    , _camera(camera)
    , _descriptor(descriptor)
{
    if (_descriptor.sceneSize.size() != 3)
        PLUGIN_THROW(
            std::runtime_error("Inalid scene size (3 floats expected"));
    _sceneSize = Vector3f(_descriptor.sceneSize[0], _descriptor.sceneSize[1],
                          _descriptor.sceneSize[2]);
    _brickSize = _sceneSize / _descriptor.nbBricks;

    PLUGIN_INFO << "=================================" << std::endl;
    PLUGIN_INFO << "Out-Of-Core engine is now enabled" << std::endl;
    PLUGIN_INFO << "Update frequency: " << _descriptor.updateFrequency
                << std::endl;
    PLUGIN_INFO << "Scene size      : " << _sceneSize << std::endl;
    PLUGIN_INFO << "Brick size      : " << _brickSize << std::endl;
    PLUGIN_INFO << "Nb of bricks    : " << _descriptor.nbBricks << std::endl;
    PLUGIN_INFO << "Visible bricks  : " << _descriptor.visibleBricks
                << std::endl;
    PLUGIN_INFO << "---------------------------------" << std::endl;
}

void OOCManager::updateBricks() {}

void OOCManager::loadBricks()
{
    GeneralSettings::getInstance()->setModelVisibilityOnCreation(false);
    std::thread t(&OOCManager::_loadBricks, this);
    t.detach();
}

void OOCManager::_loadBricks()
{
    const uint32_t numBricks = _descriptor.nbBricks;
    const int32_t visBricks = _descriptor.visibleBricks;

    std::set<int32_t> loadedBricks;
    std::set<int32_t> bricksToLoad;
    std::vector<ModelDescriptorPtr> modelsToAddToScene;
    std::vector<ModelDescriptorPtr> modelsToRemoveFromScene;
    std::vector<ModelDescriptorPtr> modelsToShow;
    int32_t previousBrickId{std::numeric_limits<int32_t>::max()};

    const auto bricksFolder = GeneralSettings::getInstance()->getBricksFolder();
    const auto dbConnectionString =
        GeneralSettings::getInstance()->getDatabaseConnectionString();
    const auto dbSchema = GeneralSettings::getInstance()->getDatabaseSchema();

    while (true)
    {
        const Vector3f cameraPosition = _camera.getPosition();
        const Vector3i brick = cameraPosition / _brickSize;

        const int32_t brickId =
            brick.z + brick.y * numBricks + brick.x * numBricks * numBricks;

        if (previousBrickId != brickId)
            bricksToLoad.clear();

        // Identify visible bricks (the ones surrounding the camera)
        std::set<int32_t> visibleBricks;
        for (int32_t x = -visBricks; x < visBricks; ++x)
            for (int32_t y = -visBricks; y < visBricks; ++y)
                for (int32_t z = -visBricks; z < visBricks; ++z)
                    visibleBricks.insert((brick.z + z) +
                                         (brick.y + y) * numBricks +
                                         (brick.x + x) * numBricks * numBricks);

        // Identify bricks to load
        for (const int32_t visibleBrick : visibleBricks)
            if (std::find(loadedBricks.begin(), loadedBricks.end(),
                          visibleBrick) == loadedBricks.end())
                bricksToLoad.insert(visibleBrick);

        if (!bricksToLoad.empty())
            PLUGIN_DEBUG << "Bricks to load : "
                         << int32_set_to_string(bricksToLoad) << std::endl;

        // Loading bricks
        if (!bricksToLoad.empty())
        {
            const auto brickToLoad = (*bricksToLoad.begin());
            loadedBricks.insert(brickToLoad);
            try
            {
                CacheLoader loader(_scene);
#ifdef USE_PQXX
                modelsToAddToScene =
                    loader.importBrickFromDB(dbConnectionString, dbSchema,
                                             brickToLoad);
#else
                char idStr[7];
                sprintf(idStr, "%06d", brickToLoad);
                const std::string filename =
                    bricksFolder + "/brick" + idStr + ".bioexplorer";
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
            PLUGIN_INFO << "Current brick Id: " << brickId << std::endl;
            PLUGIN_INFO << "Loaded bricks   : "
                        << int32_set_to_string(loadedBricks) << std::endl;
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
                        std::find(visibleBricks.begin(), visibleBricks.end(),
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

            // Prevent invisible bricks from being loaded
            auto i = loadedBricks.begin();
            while (i != loadedBricks.end())
            {
                const auto loadedBrick = (*i);
                const auto it = std::find(visibleBricks.begin(),
                                          visibleBricks.end(), loadedBrick);
                if (it == visibleBricks.end())
                    loadedBricks.erase(i++);
                else
                    ++i;
            }
        }

        bool sceneModified = false;
        for (auto md : modelsToRemoveFromScene)
        {
            PLUGIN_DEBUG << "Removing model: " << md->getModelID() << std::endl;
            _scene.removeModel(md->getModelID());
            sceneModified = true;
        }
        modelsToRemoveFromScene.clear();

        for (auto md : modelsToAddToScene)
        {
            md->setVisible(true);
            _scene.addModel(md);
            PLUGIN_DEBUG << "Adding model: " << md->getModelID() << std::endl;
            sleep(_descriptor.updateFrequency);
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

        previousBrickId = brickId;
        sleep(_descriptor.updateFrequency);
    }
}
} // namespace bioexplorer
