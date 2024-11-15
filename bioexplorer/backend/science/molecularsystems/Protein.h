/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <science/molecularsystems/Molecule.h>

namespace bioexplorer
{
namespace molecularsystems
{
/**
 * @brief The Protein class
 */
class Protein : public Molecule
{
public:
    /**
     * @brief Construct a new Protein object
     *
     * @param scene Scene to which assembly should be added
     * @param details Details of the protein
     */
    Protein(core::Scene& scene, const details::ProteinDetails& details);

    /**
     * @brief Destroy the Protein object
     *
     */
    ~Protein();

    double getTransMembraneOffset() const { return _transMembraneOffset; }
    double getTransMembraneRadius() const { return _transMembraneRadius; }

    /**
     * @brief Set the Color Scheme object
     *
     * @param colorScheme Color scheme to apply to the protein
     * @param palette Palette of colors
     * @param chainIds Optional identifiers of chains to which the color scheme
     * is to be applied
     */
    void setColorScheme(const details::ProteinColorScheme& colorScheme, const Palette& palette,
                        const size_ts& chainIds);

    /**
     * @brief Set the Amino Acid Sequence As String object
     *
     * @param aminoAcidSequence Sequence of amino acids
     */
    void setAminoAcidSequenceAsString(const std::string& aminoAcidSequence)
    {
        _selectedAminoAcidSequence = aminoAcidSequence;
        _selectedAminoAcidRanges = {{0, 0}};
    }

    /**
     * @brief Set the Amino Acid Sequence As Range object
     *
     * @param range Range of indices in the amino acids sequence
     */
    void setAminoAcidSequenceAsRanges(const Vector2uis& ranges)
    {
        _selectedAminoAcidSequence = "";
        _selectedAminoAcidRanges = ranges;
    }

    /**
     * @brief Get the protein descriptor
     *
     * @return The protein descriptor object
     */
    const details::ProteinDetails& getDescriptor() const { return _details; }

    /**
     * @brief Get the positions and rotations of Glycosylation sites on the
     * protein
     *
     * @param positions Positions of Glycosylation sites on the protein
     * @param rotations rotations of Glycosylation sites on the protein
     * @param siteIndices Optional indices of sites for which positions and
     * rotations should be returned. If empty, positions and rotations are
     * returned for every glycosylation site on the protein
     */
    void getGlycosylationSites(Vector3ds& positions, Quaternions& rotations, const size_ts& siteIndices) const;

    /**
     * @brief Get the sugar binding sites positions and rotations
     *
     * @param positions Positions of sugar binding sites on the protein
     * @param rotations rotations of sugar binding sites on the protein
     * @param siteIndices Optional indices of sites for which positions and
     * rotations should be returned. If empty, positions and rotations are
     * returned for every sugar binding site on the protein
     * @param chainIds Optional identifiers of chains for which positions and
     * rotations should be returned. If empty, positions and rotations are
     * returned for every sugar binding site on the protein
     */
    void getSugarBindingSites(Vector3ds& positions, Quaternions& rotations, const size_ts& siteIndices,
                              const size_ts& chainIds) const;

    /**
     * @brief Get the glycosylation sites of the protein
     *
     * @param siteIndices Optional indices of sites for which glycosylation
     * sites should be returned. If empty, all sites are returned
     * @return Glycosylation sites of the protein
     */
    const std::map<std::string, size_ts> getGlycosylationSites(const size_ts& siteIndices) const;

    /**
     * @brief Set an amino acid at a given position in the protein sequences
     *
     * @param details Structure containing the information related the amino
     * acid to be modified
     */
    void setAminoAcid(const details::AminoAcidDetails& details);

    /**
     * @brief addGlycan Add glycans to Glycosylation sites of a given protein
     * in the assembly
     * @param details Details of the glycans
     */
    void addGlycan(const details::SugarDetails& details);

    /**
     * @brief addSugar Add sugars to sugar binding sites of a given protein of
     * the assembly
     * @param details Details of the sugars
     */
    void addSugar(const details::SugarDetails& details);

    /**
     * @brief Get the protein transformation
     *
     * @return Transformation Protein transformation
     */
    core::Transformation getTransformation() const;

    /**
     * @brief Get the protein animation details
     *
     * @return MolecularSystemAnimationDetails Protein animation details
     */
    details::MolecularSystemAnimationDetails getAnimationDetails() const;

private:
    // Analysis
    void _getSitesTransformations(Vector3ds& positions, Quaternions& rotations,
                                  const std::map<std::string, size_ts>& sites) const;

    // Color schemes
    void _setRegionColorScheme(const Palette& palette, const size_ts& chainIds);
    void _setGlycosylationSiteColorScheme(const Palette& palette);

    // Utility functions
    void _processInstances(core::ModelDescriptorPtr md, const Vector3ds& positions, const Quaternions& rotations,
                           const core::Quaterniond& proteinRotation,
                           const details::MolecularSystemAnimationDetails& randInfo);
    void _buildAminoAcidBounds();

    // Class members
    details::ProteinDetails _details;
    GlycansMap _glycans;
    double _transMembraneOffset{0.f};
    double _transMembraneRadius;
    std::map<std::string, std::map<size_t, core::Boxd>> _aminoAcidBounds;
};
} // namespace molecularsystems
} // namespace bioexplorer
