
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_OctreeNode.h:

Program Listing for File OctreeNode.h
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_OctreeNode.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/fields/OctreeNode.h``)

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
   
   #include <brayns/common/types.h>
   
   namespace bioexplorer
   {
   namespace fields
   {
   class OctreeNode
   {
   public:
       OctreeNode(const glm::vec3 center, const float size);
   
       void addValue(const float value);
   
       void setChild(OctreeNode* child);
   
       const std::vector<OctreeNode*>& getChildren() const;
   
       const glm::vec3& getCenter() const;
   
       const float getValue() const;
   
   private:
       float _value;
   
       glm::vec3 _center;
       glm::vec3 _size;
   
       std::vector<OctreeNode*> _children;
   };
   } // namespace fields
   } // namespace bioexplorer
