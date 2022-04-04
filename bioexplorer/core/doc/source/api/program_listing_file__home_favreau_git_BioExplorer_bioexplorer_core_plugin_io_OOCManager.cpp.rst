
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_OOCManager.cpp:

Program Listing for File OOCManager.cpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_OOCManager.cpp>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/io/OOCManager.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
    * scientific data from visualization
    *
    * Copyright 2020-2022 Blue BrainProject / EPFL
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
   namespace io
   {
   using namespace common;
   
   OOCManager::OOCManager(Scene& scene, const Camera& camera,
                          const CommandLineArguments& arguments)
       : _scene(scene)
       , _camera(camera)
   {
       _parseArguments(arguments);
   
       PLUGIN_INFO("=================================");
       PLUGIN_INFO("Out-Of-Core engine is now enabled");
       PLUGIN_INFO("---------------------------------");
       PLUGIN_INFO("DB Connection string: " << _dbConnectionString);
       PLUGIN_INFO("DB Schema           : " << _dbSchema);
       PLUGIN_INFO("Description         : " << _sceneConfiguration.description);
       PLUGIN_INFO("Update frequency    : " << _updateFrequency);
       PLUGIN_INFO("Scene size          : " << _sceneConfiguration.sceneSize);
       PLUGIN_INFO("Brick size          : " << _sceneConfiguration.brickSize);
       PLUGIN_INFO("Nb of bricks        : " << _sceneConfiguration.nbBricks);
       PLUGIN_INFO("Visible bricks      : " << _nbVisibleBricks);
       PLUGIN_INFO("Bricks per cycle    : " << _nbBricksPerCycle);
       PLUGIN_INFO("Unload bricks       : " << (_unloadBricks ? "On" : "Off"));
       PLUGIN_INFO("=================================");
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
   
       uint32_t nbLoads = 0;
       float totalLoadingTime = 0.f;
   
       while (true)
       {
           const Vector3f& cameraPosition = _camera.getPosition();
           const Vector3i brick =
               (cameraPosition - _sceneConfiguration.brickSize / 2.f) /
               _sceneConfiguration.brickSize;
           const int32_t brickId = brick.z +
                                   brick.y * _sceneConfiguration.nbBricks +
                                   brick.x * _sceneConfiguration.nbBricks *
                                       _sceneConfiguration.nbBricks;
   
           if (_frameBuffer && _frameBuffer->getAccumFrames() > 1)
           {
               bricksToLoad.clear();
   
               // Identify visible bricks (the ones surrounding the camera)
               std::set<int32_t> visibleBricks;
               for (int32_t x = 0; x < _nbVisibleBricks; ++x)
                   for (int32_t y = 0; y < _nbVisibleBricks; ++y)
                       for (int32_t z = 0; z < _nbVisibleBricks; ++z)
                       {
                           visibleBricks.insert(
                               (z + brick.z) +
                               (y + brick.y) * _sceneConfiguration.nbBricks +
                               (x + brick.x) * _sceneConfiguration.nbBricks *
                                   _sceneConfiguration.nbBricks);
                           visibleBricks.insert(
                               (-z + brick.z) +
                               (-y + brick.y) * _sceneConfiguration.nbBricks +
                               (-x + brick.x) * _sceneConfiguration.nbBricks *
                                   _sceneConfiguration.nbBricks);
                       }
   
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
                   PLUGIN_INFO("Loading bricks   "
                               << int32_set_to_string(bricksToLoad));
   
               _progress = float(bricksToLoad.size()) / float(_nbBricksPerCycle);
               // Loading bricks
               if (!bricksToLoad.empty())
               {
                   const auto brickToLoad = (*bricksToLoad.begin());
                   loadedBricks.insert(brickToLoad);
                   const auto start = std::chrono::steady_clock::now();
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
                       PLUGIN_DEBUG(e.what());
                   }
                   const auto duration =
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start);
   
                   totalLoadingTime += duration.count();
                   ++nbLoads;
   
                   bricksToLoad.erase(bricksToLoad.begin());
               }
   
               if (bricksToLoad.size())
               {
                   PLUGIN_DEBUG("Current brick Id: " << brickId);
                   PLUGIN_DEBUG(
                       "Loaded bricks   : " << int32_set_to_string(loadedBricks));
                   PLUGIN_DEBUG(
                       "Visible bricks  : " << int32_set_to_string(visibleBricks));
   
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
       _sceneConfiguration = connector.getSceneConfiguration();
   
       const bool disableBroadcasting =
           std::getenv(ENV_ROCKETS_DISABLE_SCENE_BROADCASTING.c_str()) != nullptr;
       if (!disableBroadcasting)
           PLUGIN_THROW(
               ENV_ROCKETS_DISABLE_SCENE_BROADCASTING +
               " environment variable must be set when out-of-core is enabled");
   }
   } // namespace io
   } // namespace bioexplorer
