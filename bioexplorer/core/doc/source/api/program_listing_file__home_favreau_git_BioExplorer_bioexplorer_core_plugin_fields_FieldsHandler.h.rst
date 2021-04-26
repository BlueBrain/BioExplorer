
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_FieldsHandler.h:

Program Listing for File FieldsHandler.h
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_FieldsHandler.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/fields/FieldsHandler.h``)

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
   
   #include <brayns/common/simulation/AbstractSimulationHandler.h>
   
   #include <brayns/common/types.h>
   #include <brayns/engineapi/Scene.h>
   
   namespace bioexplorer
   {
   namespace fields
   {
   using namespace brayns;
   
   class FieldsHandler : public AbstractSimulationHandler
   {
   public:
       FieldsHandler(const Scene& scene, const float voxelSize,
                     const float density);
   
       FieldsHandler(const std::string& filename);
   
       FieldsHandler(const FieldsHandler& rhs);
   
       ~FieldsHandler();
   
       void* getFrameData(const uint32_t) final;
   
       bool isReady() const final { return true; }
   
       AbstractSimulationHandlerPtr clone() const final;
   
       const void exportToFile(const std::string& filename) const;
   
       void importFromFile(const std::string& filename);
   
       const glm::uvec3& getDimensions() const { return _dimensions; }
   
       const glm::vec3& getSpacing() const { return _spacing; }
   
       const glm::vec3& getOffset() const { return _offset; }
   
   private:
       void _buildOctree(const Scene& scene, const float voxelSize,
                         const float density);
   
       glm::uvec3 _dimensions;
       glm::vec3 _spacing;
       glm::vec3 _offset;
   };
   typedef std::shared_ptr<FieldsHandler> FieldsHandlerPtr;
   } // namespace fields
   } // namespace bioexplorer
