
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_OctreeNode.cpp:

Program Listing for File OctreeNode.cpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_OctreeNode.cpp>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/fields/OctreeNode.cpp``)

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
   
   #include "OctreeNode.h"
   
   namespace bioexplorer
   {
   namespace fields
   {
   OctreeNode::OctreeNode(const glm::vec3 center, const float size)
       : _value(0)
       , _center(center)
       , _size(size)
   {
   }
   
   void OctreeNode::setChild(OctreeNode* child)
   {
       _children.push_back(child);
   }
   
   void OctreeNode::addValue(float value)
   {
       if (value > _value)
           _value = value;
   }
   
   const glm::vec3& OctreeNode::getCenter() const
   {
       return _center;
   }
   
   const float OctreeNode::getValue() const
   {
       return _value;
   }
   
   const std::vector<OctreeNode*>& OctreeNode::getChildren() const
   {
       return _children;
   }
   } // namespace fields
   } // namespace bioexplorer
