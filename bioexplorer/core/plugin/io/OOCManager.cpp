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
    PLUGIN_INFO << "Bricks folder   : " << _descriptor.bricksFolder
                << std::endl;
    PLUGIN_INFO << "Scene size      : " << _sceneSize << std::endl;
    PLUGIN_INFO << "Brick size      : " << _brickSize << std::endl;
    PLUGIN_INFO << "Nb of bricks    : " << _descriptor.nbBricks << std::endl;
    PLUGIN_INFO << "Visible bricks  : " << _descriptor.visibleBricks
                << std::endl;
    PLUGIN_INFO << "---------------------------------" << std::endl;
}

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
    while (true)
    {
        const Vector3f position = _camera.getPosition();
        const Vector3f box = position / _brickSize;

        const int32_t boxId =
            box.z + box.y * numBricks + box.x * numBricks * numBricks;

        if (_currentBoxId != boxId)
            _bricksToLoad.clear();

        // Identify visible bricks (the ones surrounding the camera)
        std::set<int32_t> visibleBricks;
        for (int32_t x = -visBricks; x < visBricks; ++x)
            for (int32_t y = -visBricks; y < visBricks; ++y)
                for (int32_t z = -visBricks; z < visBricks; ++z)
                    visibleBricks.insert((box.z + z) + (box.y + y) * numBricks +
                                         (box.x + x) * numBricks * numBricks);

        // Identify bricks to load
        for (const int32_t visibleBrick : visibleBricks)
            if (std::find(_loadedBricks.begin(), _loadedBricks.end(),
                          visibleBrick) == _loadedBricks.end())
                _bricksToLoad.insert(visibleBrick);

        if (!_bricksToLoad.empty())
            PLUGIN_INFO << "Bricks to load : "
                        << int32_set_to_string(_bricksToLoad) << std::endl;

        // Loading bricks
        bool sceneModified = false;
        if (!_bricksToLoad.empty())
        {
            const auto brickToLoad = (*_bricksToLoad.begin());
            _loadedBricks.insert(brickToLoad);
            char idStr[7];
            sprintf(idStr, "%06d", brickToLoad);
            const std::string filename =
                _descriptor.bricksFolder + "/brick" + idStr + ".bioexplorer";
            try
            {
                CacheLoader loader(_scene, brickToLoad);
                const auto modelDescriptors =
                    loader.importModelsFromFile(filename);

                if (!modelDescriptors.empty())
                {
                    for (const auto modelDescriptor : modelDescriptors)
                        _scene.addModel(modelDescriptor);
                    // Wait for the model BVH to be built
                    sleep(_descriptor.updateFrequency);
                    sceneModified = true;
                }
            }
            catch (std::runtime_error& e)
            {
                PLUGIN_INFO << e.what() << std::endl;
            }
            _bricksToLoad.erase(brickToLoad);
        }

        if (_bricksToLoad.size() % 5 == 0 && sceneModified)
        {
            PLUGIN_DEBUG << "Current box Id: " << boxId << std::endl;
            PLUGIN_DEBUG << "Visible bricks: "
                         << int32_set_to_string(visibleBricks) << std::endl;
            PLUGIN_DEBUG << "Loaded bricks : "
                         << int32_set_to_string(_loadedBricks) << std::endl;

            bool visibilityModified = false;

            // Make visible models visible and remove invisible models
            auto modelDescriptors = _scene.getModelDescriptors();
            for (auto modelDescriptor : modelDescriptors)
            {
                const auto metadata = modelDescriptor->getMetadata();
                const auto it = metadata.find("boxid");
                if (it != metadata.end())
                {
                    const int32_t id = atoi(it->second.c_str());
                    const bool visible =
                        std::find(visibleBricks.begin(), visibleBricks.end(),
                                  id) != visibleBricks.end();
                    if (visible)
                    {
                        if (!modelDescriptor->getVisible())
                        {
                            modelDescriptor->setVisible(true);
                            visibilityModified = true;
                        }
                    }
                    else
                        _scene.removeModel(modelDescriptor->getModelID());
                }
            }

            PLUGIN_DEBUG << "Removing invisible bricks" << std::endl;
            for (const auto loadedBrick : _loadedBricks)
            {
                auto it = std::find(visibleBricks.begin(), visibleBricks.end(),
                                    loadedBrick);
                if (it == visibleBricks.end())
                    _loadedBricks.erase(loadedBrick);
            }

            if (visibilityModified)
            {
                PLUGIN_INFO << "Scene has been modified" << std::endl;
                _scene.markModified(false);
            }
            else
            {
                PLUGIN_INFO << "Scene has been unmodified" << std::endl;
                _scene.resetModified();
            }
        }
        _currentBoxId = boxId;

        // std::this_thread::yield();
    }
}
} // namespace bioexplorer
