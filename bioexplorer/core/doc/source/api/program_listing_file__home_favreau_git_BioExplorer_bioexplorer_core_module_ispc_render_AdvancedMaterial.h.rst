
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_module_ispc_render_AdvancedMaterial.h:

Program Listing for File AdvancedMaterial.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_module_ispc_render_AdvancedMaterial.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/module/ispc/render/AdvancedMaterial.h``)

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
   
   #include <plugin/common/CommonTypes.h>
   
   #include <ospray/SDK/common/Material.h>
   #include <ospray/SDK/texture/Texture2D.h>
   
   namespace bioexplorer
   {
   namespace rendering
   {
   typedef ospray::vec3f Color;
   
   struct AdvancedMaterial : public ospray::Material
   {
       ospray::Texture2D* map_d;
       ospray::affine2f xform_d;
       float d;
   
       ospray::Texture2D* map_Refraction;
       ospray::affine2f xform_Refraction;
       float refraction;
   
       ospray::Texture2D* map_Reflection;
       ospray::affine2f xform_Reflection;
       float reflection;
   
       ospray::Texture2D* map_a;
       ospray::affine2f xform_a;
       float a;
   
       ospray::Texture2D* map_Kd;
       ospray::affine2f xform_Kd;
       Color Kd;
   
       ospray::Texture2D* map_Ks;
       ospray::affine2f xform_Ks;
       Color Ks;
   
       ospray::Texture2D* map_Ns;
       ospray::affine2f xform_Ns;
       float Ns;
   
       float glossiness;
   
       ospray::Texture2D* map_Bump;
       ospray::affine2f xform_Bump;
       ospray::linear2f rot_Bump;
   
       MaterialShadingMode shadingMode;
   
       float userParameter;
   
       ospray::uint32 nodeId;
   
       MaterialChameleonMode chameleonMode;
   
       std::string toString() const final { return "default_material"; }
       void commit() final;
   };
   } // namespace rendering
   } // namespace bioexplorer
