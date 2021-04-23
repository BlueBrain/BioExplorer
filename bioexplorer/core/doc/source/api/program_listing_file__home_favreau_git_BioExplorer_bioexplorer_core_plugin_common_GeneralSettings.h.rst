
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_GeneralSettings.h:

Program Listing for File GeneralSettings.h
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_GeneralSettings.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/common/GeneralSettings.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

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
   namespace common
   {
   class GeneralSettings
   {
   public:
       GeneralSettings() {}
   
       static GeneralSettings* getInstance()
       {
           std::lock_guard<std::mutex> lock(_mutex);
           if (!_instance)
               _instance = new GeneralSettings();
           return _instance;
       }
   
       bool getModelVisibilityOnCreation() { return _modelVisibilityOnCreation; }
   
       void setModelVisibilityOnCreation(const bool value)
       {
           _modelVisibilityOnCreation = value;
       }
   
       std::string getOffFolder() { return _offFolder; }
   
       void setOffFolder(const std::string& value) { _offFolder = value; }
   
       bool getLoggingEnabled() const { return _loggingEnabled; }
   
       void setLoggingEnabled(const bool value) { _loggingEnabled = value; }
   
       static std::mutex _mutex;
       static GeneralSettings* _instance;
   
   private:
       ~GeneralSettings() {}
   
       bool _modelVisibilityOnCreation{true};
       std::string _offFolder{"/tmp/"};
       bool _loggingEnabled{false};
   };
   } // namespace common
   } // namespace bioexplorer
