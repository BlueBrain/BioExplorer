
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_Octree.cpp:

Program Listing for File Octree.cpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_fields_Octree.cpp>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/fields/Octree.cpp``)

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
   
   #include "Octree.h"
   
   #include <plugin/common/Logs.h>
   
   namespace bioexplorer
   {
   namespace fields
   {
   using namespace std;
   
   typedef std::map<uint64_t, OctreeNode> OctreeLevelMap;
   
   Octree::Octree(const floats &events, float voxelSize, const glm::vec3 &minAABB,
                  const glm::vec3 &maxAABB)
       : _volumeDim(glm::uvec3(0u, 0u, 0u))
       , _volumeSize(0u)
       , _offsetPerLevel(nullptr)
   {
       PLUGIN_INFO("Nb of events : " << events.size() / 5);
   
       // **************** Octree creations *******************
       // *****************************************************
       glm::uvec3 octreeSize(
           _pow2roundup(std::ceil((maxAABB.x - minAABB.x) / voxelSize)),
           _pow2roundup(std::ceil((maxAABB.y - minAABB.y) / voxelSize)),
           _pow2roundup(std::ceil((maxAABB.z - minAABB.z) / voxelSize)));
   
       // This octree is always cubic
       _octreeSize = std::max(std::max(octreeSize.x, octreeSize.y), octreeSize.z);
   
       PLUGIN_INFO("Octree size  : " << _octreeSize);
   
       uint32_t octreeDepth = std::log2(_octreeSize) + 1u;
       std::vector<OctreeLevelMap> octree(octreeDepth);
   
       PLUGIN_INFO("Octree depth : " << octreeDepth << " " << octree.size());
   
       if (octreeDepth == 0)
           return;
   
       for (uint32_t i = 0; i < events.size(); i += 5)
       {
           const uint64_t xpos = std::floor((events[i] - minAABB.x) / voxelSize);
           const uint64_t ypos =
               std::floor((events[i + 1] - minAABB.y) / voxelSize);
           const uint64_t zpos =
               std::floor((events[i + 2] - minAABB.z) / voxelSize);
           const float value = events[i + 4];
   
           const uint64_t indexX = xpos;
           const uint64_t indexY = ypos * (uint64_t)_octreeSize;
           const uint64_t indexZ =
               zpos * (uint64_t)_octreeSize * (uint64_t)_octreeSize;
   
           auto it = octree[0].find(indexX + indexY + indexZ);
           if (it == octree[0].end())
           {
               OctreeNode *child = nullptr;
               for (uint32_t level = 0; level < octreeDepth; ++level)
               {
                   bool newNode = false;
                   const uint64_t divisor = std::pow(2, level);
                   const glm::vec3 center(divisor * (xpos / divisor + 0.5f),
                                          divisor * (ypos / divisor + 0.5f),
                                          divisor * (zpos / divisor + 0.5f));
   
                   const uint64_t nBlock = _octreeSize / divisor;
                   const uint64_t index =
                       std::floor(xpos / divisor) +
                       nBlock * std::floor(ypos / divisor) +
                       nBlock * nBlock * std::floor(zpos / divisor);
   
                   const float size = voxelSize * (level + 1u);
   
                   if (octree[level].find(index) == octree[level].end())
                   {
                       octree[level].insert(
                           OctreeLevelMap::value_type(index,
                                                      OctreeNode(center, size)));
                       newNode = true;
                   }
   
                   octree[level].at(index).addValue(value);
   
                   if ((level > 0) && (child != nullptr))
                       octree[level].at(index).setChild(child);
   
                   if (newNode)
                       child = &(octree[level].at(index));
                   else
                       child = nullptr;
               }
           }
           else
           {
               for (uint32_t level = 0; level < octreeDepth; ++level)
               {
                   const uint64_t divisor = std::pow(2, level);
                   const uint64_t nBlock = _octreeSize / divisor;
                   const uint64_t index =
                       std::floor(xpos / divisor) +
                       nBlock * std::floor(ypos / divisor) +
                       nBlock * nBlock * std::floor(zpos / divisor);
                   octree[level].at(index).addValue(value);
               }
           }
       }
       for (uint32_t i = 0; i < octree.size(); ++i)
           PLUGIN_DEBUG("Number of leaves [" << i << "]: " << octree[i].size());
   
       // **************** Octree flattening *******************
       // ******************************************************
   
       _offsetPerLevel = new uint32_t[octreeDepth];
       _offsetPerLevel[octreeDepth - 1u] = 0;
       uint32_t previousOffset = 0u;
       for (uint32_t i = octreeDepth - 1u; i > 0u; --i)
       {
           _offsetPerLevel[i - 1u] = previousOffset + octree[i].size();
           previousOffset = _offsetPerLevel[i - 1u];
       }
   
       uint32_t totalNodeNumber = 0;
   
       for (uint32_t i = 0; i < octree.size(); ++i)
           totalNodeNumber += octree[i].size();
   
       // need to be initialized with zeros
       _flatIndexes.resize(totalNodeNumber * 2u, 0);
       _flatData.resize(totalNodeNumber * 4);
   
       // The root node
       _flattenChildren(&(octree[octreeDepth - 1u].at(0)), octreeDepth - 1u);
   
       // **************** Octree flattening end *****************
       // ********************************************************
   
       _volumeDim = glm::uvec3(std::ceil((maxAABB.x - minAABB.x) / voxelSize),
                               std::ceil((maxAABB.y - minAABB.y) / voxelSize),
                               std::ceil((maxAABB.z - minAABB.z) / voxelSize));
       _volumeSize = (uint64_t)_volumeDim.x * (uint64_t)_volumeDim.y *
                     (uint64_t)_volumeDim.z;
   }
   
   Octree::~Octree()
   {
       delete[] _offsetPerLevel;
   }
   
   void Octree::_flattenChildren(const OctreeNode *node, uint32_t level)
   {
       const std::vector<OctreeNode *> children = node->getChildren();
   
       if ((children.empty()) || (level == 0))
       {
           _flatData[_offsetPerLevel[level] * 4u] = node->getCenter().x;
           _flatData[_offsetPerLevel[level] * 4u + 1] = node->getCenter().y;
           _flatData[_offsetPerLevel[level] * 4u + 2] = node->getCenter().z;
           _flatData[_offsetPerLevel[level] * 4u + 3] = node->getValue();
   
           _offsetPerLevel[level] += 1u;
           return;
       }
       _flatData[_offsetPerLevel[level] * 4u] = node->getCenter().x;
       _flatData[_offsetPerLevel[level] * 4u + 1] = node->getCenter().y;
       _flatData[_offsetPerLevel[level] * 4u + 2] = node->getCenter().z;
       _flatData[_offsetPerLevel[level] * 4u + 3] = node->getValue();
   
       _flatIndexes[_offsetPerLevel[level] * 2u] = _offsetPerLevel[level - 1];
       _flatIndexes[_offsetPerLevel[level] * 2u + 1] =
           _offsetPerLevel[level - 1] + children.size() - 1u;
       _offsetPerLevel[level] += 1u;
   
       for (const OctreeNode *child : children)
           _flattenChildren(child, level - 1u);
   }
   
   const uint32_t Octree::getOctreeSize() const
   {
       return _octreeSize;
   }
   
   const uint32_ts &Octree::getFlatIndexes() const
   {
       return _flatIndexes;
   }
   
   const floats &Octree::getFlatData() const
   {
       return _flatData;
   }
   
   const glm::uvec3 &Octree::getVolumeDim() const
   {
       return _volumeDim;
   }
   
   const uint64_t Octree::getVolumeSize() const
   {
       return _volumeSize;
   }
   } // namespace fields
   } // namespace bioexplorer
