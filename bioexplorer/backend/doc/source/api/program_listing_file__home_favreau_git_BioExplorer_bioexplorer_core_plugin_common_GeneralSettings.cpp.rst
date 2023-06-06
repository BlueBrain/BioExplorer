
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_GeneralSettings.cpp:

Program Listing for File GeneralSettings.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_GeneralSettings.cpp>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/common/GeneralSettings.cpp``)

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
   
   #include "GeneralSettings.h"
   
   namespace bioexplorer
   {
   namespace common
   {
   GeneralSettings *GeneralSettings::_instance = nullptr;
   std::mutex GeneralSettings::_mutex;
   } // namespace common
   } // namespace bioexplorer
