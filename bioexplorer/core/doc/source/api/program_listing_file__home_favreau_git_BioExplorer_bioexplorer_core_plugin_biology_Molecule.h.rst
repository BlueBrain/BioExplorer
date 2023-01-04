
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Molecule.h:

Program Listing for File Molecule.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Molecule.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/biology/Molecule.h``)

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
   
   #include <plugin/biology/Node.h>
   
   #include <brayns/engineapi/Model.h>
   
   namespace bioexplorer
   {
   namespace biology
   {
   using namespace brayns;
   using namespace details;
   
   const std::string KEY_UNDEFINED = "Undefined";
   const std::string KEY_ATOM = "ATOM";
   const std::string KEY_HETATM = "HETATM";
   const std::string KEY_HEADER = "HEADER";
   const std::string KEY_TITLE = "TITLE";
   const std::string KEY_CONECT = "CONECT";
   const std::string KEY_SEQRES = "SEQRES";
   const std::string KEY_REMARK = "REMARK";
   
   class Molecule : public Node
   {
   public:
       Molecule(Scene& scene, const size_ts& chainIds);
   
       const AtomMap& getAtoms() const { return _atomMap; }
   
       const Residues& getResidues() const { return _residues; }
   
       const ResidueSequenceMap& getResidueSequences() const
       {
           return _residueSequenceMap;
       }
   
       const StringMap getSequencesAsString() const;
   
   protected:
       void _setAtomColorScheme();
       void _setChainColorScheme(const Palette& palette);
       void _setResiduesColorScheme(const Palette& palette);
       void _setAminoAcidSequenceColorScheme(const Palette& palette);
       void _setMaterialDiffuseColor(const size_t atomIndex,
                                     const RGBColorDetails& color);
       void _setMaterialDiffuseColor(const size_t atomIndex, const Color& color);
   
       // Geometry
       void _buildModel(const std::string& assemblyName, const std::string& name,
                        const std::string& title, const std::string& header,
                        const ProteinRepresentation& representation,
                        const float atomRadiusMultiplier, const bool loadBonds);
   
       void _buildAtomicStruture(const ProteinRepresentation representation,
                                 const float atomRadiusMultiplier,
                                 const bool surface, const bool loadBonds,
                                 Model& model);
       void _computeReqSetOffset();
   
       // IO
       void _readAtom(const std::string& line, const bool loadHydrogen);
       void _readSequence(const std::string& line);
       std::string _readHeader(const std::string& line);
       std::string _readTitle(const std::string& line);
       void _readRemark(const std::string& line);
       void _readConnect(const std::string& line);
       bool _loadChain(const size_t chainId);
   
       Scene& _scene;
       AtomMap _atomMap;
       Residues _residues;
       ResidueSequenceMap _residueSequenceMap;
       BondsMap _bondsMap;
       size_ts _chainIds;
   
       Vector2ui _aminoAcidRange;
   
       std::string _selectedAminoAcidSequence;
       Vector2uis _selectedAminoAcidRanges;
       Boxf _bounds;
   };
   } // namespace biology
   } // namespace bioexplorer
