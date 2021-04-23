
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Assembly.h:

Program Listing for File Assembly.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Assembly.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/biology/Assembly.h``)

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
   
   #include <plugin/common/Types.h>
   
   namespace bioexplorer
   {
   namespace biology
   {
   using namespace details;
   
   class Assembly
   {
   public:
       Assembly(Scene &scene, const AssemblyDetails &details);
   
       ~Assembly();
   
       void setColorScheme(const ColorSchemeDetails &details);
   
       void setAminoAcidSequenceAsString(
           const AminoAcidSequenceAsStringDetails &details);
   
       void setAminoAcidSequenceAsRange(
           const AminoAcidSequenceAsRangesDetails &details);
   
       const std::string getAminoAcidInformation(
           const AminoAcidInformationDetails &details) const;
   
       void setAminoAcid(const AminoAcidDetails &details);
   
       Vector4fs &getClippingPlanes() { return _clippingPlanes; }
   
       void setClippingPlanes(const Vector4fs &clippingPlanes)
       {
           _clippingPlanes = clippingPlanes;
       }
   
       const AssemblyDetails &getDescriptor() { return _details; }
   
       const ProteinMap &getProteins() const { return _proteins; }
   
       void setProteinInstanceTransformation(
           const ProteinInstanceTransformationDetails &details);
   
       const Transformation getProteinInstanceTransformation(
           const ProteinInstanceTransformationDetails &details) const;
   
       void addMembrane(const MembraneDetails &details);
   
       void addRNASequence(const RNASequenceDetails &details);
   
       void addProtein(const ProteinDetails &details);
   
       void addMeshBasedMembrane(const MeshBasedMembraneDetails &details);
   
       void addGlycans(const SugarsDetails &details);
   
       void addSugars(const SugarsDetails &details);
   
   private:
       void _processInstances(ModelDescriptorPtr md, const std::string &name,
                              const AssemblyShape shape,
                              const floats &assemblyParams,
                              const size_t occurrences, const Vector3f &position,
                              const Quaterniond &orientation,
                              const size_ts &allowedOccurrences,
                              const size_t randomSeed,
                              const PositionRandomizationType &randomizationType);
   
       AssemblyDetails _details;
       Scene &_scene;
       ProteinMap _proteins;
       MeshBasedMembraneMap _meshBasedMembranes;
       MembranePtr _membrane{nullptr};
       RNASequencePtr _rnaSequence{nullptr};
       Vector4fs _clippingPlanes;
   };
   } // namespace biology
   } // namespace bioexplorer
