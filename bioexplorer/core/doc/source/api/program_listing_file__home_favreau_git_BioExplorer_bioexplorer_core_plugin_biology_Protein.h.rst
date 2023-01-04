
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Protein.h:

Program Listing for File Protein.h
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Protein.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/biology/Protein.h``)

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
   
   #include <plugin/biology/Molecule.h>
   
   namespace bioexplorer
   {
   namespace biology
   {
   class Protein : public Molecule
   {
   public:
       Protein(Scene& scene, const ProteinDetails& details);
   
       ~Protein();
   
       void setColorScheme(const ColorScheme& colorScheme, const Palette& palette,
                           const size_ts& chainIds);
   
       void setAminoAcidSequenceAsString(const std::string& aminoAcidSequence)
       {
           _selectedAminoAcidSequence = aminoAcidSequence;
           _selectedAminoAcidRanges = {{0, 0}};
       }
   
       void setAminoAcidSequenceAsRanges(const Vector2uis& ranges)
       {
           _selectedAminoAcidSequence = "";
           _selectedAminoAcidRanges = ranges;
       }
   
       const ProteinDetails& getDescriptor() const { return _details; }
   
       void getGlycosilationSites(Vector3fs& positions, Quaternions& rotations,
                                  const size_ts& siteIndices) const;
   
       void getSugarBindingSites(Vector3fs& positions, Quaternions& rotations,
                                 const size_ts& siteIndices,
                                 const size_ts& chainIds) const;
   
       const std::map<std::string, size_ts> getGlycosylationSites(
           const size_ts& siteIndices) const;
   
       void setAminoAcid(const AminoAcidDetails& details);
   
       void addGlycans(const SugarsDetails& details);
   
       void addSugars(const SugarsDetails& details);
   
   private:
       // Analysis
       void _getSitesTransformations(
           Vector3fs& positions, Quaternions& rotations,
           const std::map<std::string, size_ts>& sites) const;
   
       // Color schemes
       void _setRegionColorScheme(const Palette& palette, const size_ts& chainIds);
       void _setGlycosylationSiteColorScheme(const Palette& palette);
   
       // Utility functions
       void _processInstances(ModelDescriptorPtr md, const Vector3fs& positions,
                              const Quaternions& rotations,
                              const Quaterniond& proteinrotation);
       void _buildAminoAcidBounds();
   
       // Class members
       ProteinDetails _details;
       GlycansMap _glycans;
       std::map<std::string, std::map<size_t, Boxf>> _aminoAcidBounds;
   };
   } // namespace biology
   } // namespace bioexplorer
