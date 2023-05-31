
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_CommonTypes.h:

Program Listing for File CommonTypes.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_CommonTypes.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/common/CommonTypes.h``)

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
   
   #define DEFAULT_SKY_POWER 4.f
   
   #define MATERIAL_PROPERTY_SHADING_MODE "shading_mode"
   #define MATERIAL_PROPERTY_USER_PARAMETER "user_parameter"
   #define MATERIAL_PROPERTY_CHAMELEON_MODE "chameleon_mode"
   #define MATERIAL_PROPERTY_NODE_ID "node_id"
   
   enum MaterialShadingMode
   {
       undefined_shading_mode = 0,
       basic = 1,
       diffuse = 2,
       electron = 3,
       cartoon = 4,
       electron_transparency = 5,
       perlin = 6,
       diffuse_transparency = 7,
       checker = 8,
       goodsell = 9
   };
   
   enum MaterialChameleonMode
   {
       undefined_chameleon_mode = 0,
       emitter = 1,
       receiver = 2
   };
   
   enum CameraStereoMode
   {
       mono = 0,
       left = 1,
       right = 2,
       side_by_side = 3
   };
