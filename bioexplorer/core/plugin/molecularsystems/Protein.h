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

#include <plugin/molecularsystems/Molecule.h>

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
    Protein(Scene& scene, const ProteinDetails& details);

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
    void setColorScheme(const ProteinColorScheme& colorScheme,
                        const Palette& palette, const size_ts& chainIds);

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
    const ProteinDetails& getDescriptor() const { return _details; }

    /**
     * @brief Get the positions and rotations of glycosilation sites on the
     * protein
     *
     * @param positions Positions of glycosilation sites on the protein
     * @param rotations rotations of glycosilation sites on the protein
     * @param siteIndices Optional indices of sites for which positions and
     * rotations should be returned. If empty, positions and rotations are
     * returned for every glycosylation site on the protein
     */
    void getGlycosilationSites(Vector3ds& positions, Quaternions& rotations,
                               const size_ts& siteIndices) const;

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
    void getSugarBindingSites(Vector3ds& positions, Quaternions& rotations,
                              const size_ts& siteIndices,
                              const size_ts& chainIds) const;

    /**
     * @brief Get the glycosylation sites of the protein
     *
     * @param siteIndices Optional indices of sites for which glycosylation
     * sites should be returned. If empty, all sites are returned
     * @return Glycosylation sites of the protein
     */
    const std::map<std::string, size_ts> getGlycosylationSites(
        const size_ts& siteIndices) const;

    /**
     * @brief Set an amino acid at a given position in the protein sequences
     *
     * @param details Structure containing the information related the amino
     * acid to be modified
     */
    void setAminoAcid(const AminoAcidDetails& details);

    /**
     * @brief addGlycan Add glycans to glycosilation sites of a given protein
     * in the assembly
     * @param details Details of the glycans
     */
    void addGlycan(const SugarDetails& details);

    /**
     * @brief addSugar Add sugars to sugar binding sites of a given protein of
     * the assembly
     * @param details Details of the sugars
     */
    void addSugar(const SugarDetails& details);

    /**
     * @brief Get the protein transformation
     *
     * @return Transformation Protein transformation
     */
    Transformation getTransformation() const;

    /**
     * @brief Get the protein animation details
     *
     * @return MolecularSystemAnimationDetails Protein animation details
     */
    MolecularSystemAnimationDetails getAnimationDetails() const;

private:
    // Analysis
    void _getSitesTransformations(
        Vector3ds& positions, Quaternions& rotations,
        const std::map<std::string, size_ts>& sites) const;

    // Color schemes
    void _setRegionColorScheme(const Palette& palette, const size_ts& chainIds);
    void _setGlycosylationSiteColorScheme(const Palette& palette);

    // Utility functions
    void _processInstances(ModelDescriptorPtr md, const Vector3ds& positions,
                           const Quaternions& rotations,
                           const Quaterniond& proteinrotation,
                           const MolecularSystemAnimationDetails& randInfo);
    void _buildAminoAcidBounds();

    // Class members
    ProteinDetails _details;
    GlycansMap _glycans;
    double _transMembraneOffset{0.f};
    double _transMembraneRadius;
    std::map<std::string, std::map<size_t, Boxf>> _aminoAcidBounds;
};
} // namespace molecularsystems
} // namespace bioexplorer
