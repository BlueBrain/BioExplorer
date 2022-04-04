
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_module_ispc_render_AdvancedRenderer.h:

Program Listing for File AdvancedRenderer.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_module_ispc_render_AdvancedRenderer.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/module/ispc/render/AdvancedRenderer.h``)

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
   
   #pragma once
   
   #include "AdvancedMaterial.h"
   
   #include <ospray/SDK/render/Renderer.h>
   
   namespace bioexplorer
   {
   namespace rendering
   {
   class AdvancedRenderer : public ospray::Renderer
   {
   public:
       AdvancedRenderer();
   
       std::string toString() const final { return "bio_explorer_renderer"; }
   
       void commit() final;
   
   private:
       // Shading attributes
       std::vector<void*> _lightArray;
       void** _lightPtr;
       ospray::Data* _lightData;
   
       AdvancedMaterial* _bgMaterial;
   
       bool _useHardwareRandomizer{false};
       bool _showBackground{false};
   
       float _timestamp{0.f};
       float _exposure{1.f};
   
       float _fogThickness{1e6f};
       float _fogStart{0.f};
   
       ospray::uint32 _maxBounces{3};
       ospray::uint32 _randomNumber{0};
   
       float _shadows{0.f};
       float _softShadows{0.f};
       ospray::uint32 _softShadowsSamples{0};
   
       float _giStrength{0.f};
       float _giDistance{1e6f};
       ospray::uint32 _giSamples{0};
   };
   } // namespace rendering
   } // namespace bioexplorer
