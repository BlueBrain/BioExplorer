
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_OOCManager.h:

Program Listing for File OOCManager.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_OOCManager.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/io/OOCManager.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

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
   
   #pragma once
   
   #include <plugin/common/Types.h>
   
   namespace bioexplorer
   {
   namespace io
   {
   using namespace brayns;
   using namespace details;
   
   class OOCManager
   {
   public:
       OOCManager(Scene& scene, const Camera& camera,
                  const CommandLineArguments& arguments);
   
       ~OOCManager() {}
   
       void setFrameBuffer(FrameBuffer* frameBuffer)
       {
           _frameBuffer = frameBuffer;
       }
   
       const FrameBuffer* getFrameBuffer() const { return _frameBuffer; }
   
       void loadBricks();
   
       const OOCSceneConfigurationDetails& getSceneConfiguration() const
       {
           return _sceneConfiguration;
       }
   
       const bool getShowGrid() const { return _showGrid; }
   
       const int32_t getVisibleBricks() const { return _nbVisibleBricks; }
   
       const float getUpdateFrequency() const { return _updateFrequency; }
   
       const float getProgress() const { return _progress; }
   
       const float getAverageLoadingTime() const { return _averageLoadingTime; }
   
   private:
       void _parseArguments(const CommandLineArguments& arguments);
       void _loadBricks();
   
       Scene& _scene;
       const Camera& _camera;
       FrameBuffer* _frameBuffer{nullptr};
   
       OOCSceneConfigurationDetails _sceneConfiguration;
       float _updateFrequency{1.f};
       int32_t _nbVisibleBricks{0};
       bool _unloadBricks{false};
       bool _showGrid{false};
       uint32_t _nbBricksPerCycle{5};
       float _progress{0.f};
       float _averageLoadingTime{0.f};
   
       // IO
   #ifdef USE_PQXX
       std::string _dbConnectionString;
       std::string _dbSchema;
   #else
       std::string _bricksFolder;
   #endif
   };
   } // namespace io
   } // namespace bioexplorer
