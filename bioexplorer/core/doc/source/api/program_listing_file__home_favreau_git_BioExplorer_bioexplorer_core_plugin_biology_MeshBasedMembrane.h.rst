
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_MeshBasedMembrane.h:

Program Listing for File MeshBasedMembrane.h
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_MeshBasedMembrane.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/biology/MeshBasedMembrane.h``)

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
   
   #include <plugin/biology/Node.h>
   
   #include <brayns/engineapi/Model.h>
   
   #include <assimp/Importer.hpp>
   #include <assimp/postprocess.h>
   #include <assimp/scene.h>
   #include <assimp/version.h>
   
   namespace bioexplorer
   {
   namespace biology
   {
   class MeshBasedMembrane : public Node
   {
   public:
       MeshBasedMembrane(Scene& scene, const MeshBasedMembraneDetails& details);
   
       const ProteinPtr getProtein() const { return _protein; }
   
   private:
       float _getSurfaceArea(const Vector3f& a, const Vector3f& b,
                             const Vector3f& c) const;
       Vector3f _toVector3f(const aiVector3D& v) const;
       Vector3f _toVector3f(const aiVector3D& v, const Vector3f& center,
                            const Vector3f& scaling,
                            const Quaterniond& rotation) const;
   
       ProteinPtr _protein;
   };
   } // namespace biology
   } // namespace bioexplorer
