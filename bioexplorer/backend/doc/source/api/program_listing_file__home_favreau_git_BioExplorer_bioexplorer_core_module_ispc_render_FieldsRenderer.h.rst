
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_module_ispc_render_FieldsRenderer.h:

Program Listing for File FieldsRenderer.h
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_module_ispc_render_FieldsRenderer.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/module/ispc/render/FieldsRenderer.h``)

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
   
   #include "AdvancedMaterial.h"
   
   #include <ospray/SDK/render/Renderer.h>
   
   namespace bioexplorer
   {
   namespace rendering
   {
   using namespace ospray;
   
   class FieldsRenderer : public ospray::Renderer
   {
   public:
       FieldsRenderer();
   
       std::string toString() const final { return "bio_explorer_fields"; }
   
       void commit() final;
   
   private:
       // Shading attributes
       std::vector<void*> _lightArray;
       void** _lightPtr;
       ospray::Data* _lightData;
   
       AdvancedMaterial* _bgMaterial;
   
       bool _useHardwareRandomizer{false};
       ospray::uint32 _randomNumber{0};
   
       float _timestamp{0.f};
       float _exposure{1.f};
   
       float _alphaCorrection{1.f};
   
       // Octree
       float _minRayStep;
       ospray::uint32 _nbRaySteps;
       ospray::uint32 _nbRayRefinementSteps;
   
       float _cutoff;
       ospray::Ref<ospray::Data> _userData;
       ospray::uint64 _userDataSize;
   };
   } // namespace rendering
   } // namespace bioexplorer
