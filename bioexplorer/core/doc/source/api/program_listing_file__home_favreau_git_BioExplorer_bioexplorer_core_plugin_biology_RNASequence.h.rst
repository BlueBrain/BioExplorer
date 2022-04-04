
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_RNASequence.h:

Program Listing for File RNASequence.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_RNASequence.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/biology/RNASequence.h``)

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
   
   #include <plugin/biology/Node.h>
   
   #include <brayns/engineapi/Model.h>
   
   namespace bioexplorer
   {
   namespace biology
   {
   using namespace details;
   
   class RNASequence : public Node
   {
   public:
       RNASequence(Scene& scene, const RNASequenceDetails& details);
   
       RNASequenceMap getRNASequences() { return _rnaSequenceMap; }
   
   private:
       Vector3f _trefoilKnot(float R, float t, const Vector3f& params);
       Vector3f _torus(float R, float t, const Vector3f& params);
       Vector3f _star(float R, float t);
       Vector3f _spring(float R, float t);
       Vector3f _heart(float R, float u);
       Vector3f _thing(float R, float t, const Vector3f& a);
       Vector3f _moebius(float R, float u, float v);
   
       RNASequenceMap _rnaSequenceMap;
   };
   } // namespace biology
   } // namespace bioexplorer
