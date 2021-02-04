/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <plugin/api/Params.h>

#include <plugin/bioexplorer/Molecule.h>
#include <plugin/common/Types.h>

namespace bioexplorer
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
     * @param descriptor Description of the protein
     */
    Protein(Scene& scene, const ProteinDescriptor& descriptor);

    /**
     * @brief Set the Color Scheme object
     *
     * @param colorScheme Color scheme to apply to the protein
     * @param palette Palette of colors
     * @param chainIds Optional identifiers of chains to which the color scheme
     * is to be applied
     */
    void setColorScheme(const ColorScheme& colorScheme, const Palette& palette,
                        const size_ts& chainIds);

    /**
     * @brief Set the Amino Acid Sequence As String object
     *
     * @param aminoAcidSequence Sequence of amino acids
     */
    void setAminoAcidSequenceAsString(const std::string& aminoAcidSequence)
    {
        _aminoAcidSequence = aminoAcidSequence;
        _aminoAcidRange = {0, 0};
    }

    /**
     * @brief Set the Amino Acid Sequence As Range object
     *
     * @param range Range of indices in the amino acids sequence
     */
    void setAminoAcidSequenceAsRange(const Vector2ui& range)
    {
        _aminoAcidSequence = "";
        _aminoAcidRange = range;
    }

    /**
     * @brief Get the protein descriptor
     *
     * @return The protein descriptor object
     */
    const ProteinDescriptor& getDescriptor() const { return _descriptor; }

    /**
     * @brief Get the positions and rotations of glycosilation sites on the
     * protein
     *
     * @param positions Positions of glycosilation sites on the protein
     * @param orientations Orientations of glycosilation sites on the protein
     * @param siteIndices Optional indices of sites for which positions and
     * rotations should be returned. If empty, positions and rotations are
     * returned for every glycosylation site on the protein
     */
    void getGlycosilationSites(std::vector<Vector3f>& positions,
                               std::vector<Quaterniond>& orientations,
                               const std::vector<size_t>& siteIndices) const;

    /**
     * @brief Get the sugar binding sites positions and orientations
     *
     * @param positions Positions of sugar binding sites on the protein
     * @param orientations Orientations of sugar binding sites on the protein
     * @param siteIndices Optional indices of sites for which positions and
     * orientations should be returned. If empty, positions and rotations are
     * returned for every sugar binding site on the protein
     * @param chainIds Optional identifiers of chains for which positions and
     * orientations should be returned. If empty, positions and orientations are
     * returned for every sugar binding site on the protein
     */
    void getSugarBindingSites(std::vector<Vector3f>& positions,
                              std::vector<Quaterniond>& orientations,
                              const std::vector<size_t>& siteIndices,
                              const size_ts& chainIds) const;

    /**
     * @brief Get the glycosylation sites of the protein
     *
     * @param siteIndices Optional indices of sites for which glycosylation
     * sites should be returned. If empty, all sites are returned
     * @return Glycosylation sites of the protein
     */
    std::map<std::string, size_ts> getGlycosylationSites(
        const size_ts& siteIndices) const;

private:
    // Analysis
    void _getSitesTransformations(
        std::vector<Vector3f>& positions, std::vector<Quaterniond>& rotations,
        const std::map<std::string, size_ts>& sites) const;

    // Color schemes
    void _setRegionColorScheme(const Palette& palette, const size_ts& chainIds);
    void _setGlycosylationSiteColorScheme(const Palette& palette);

    // Class members
    ProteinDescriptor _descriptor;
};
} // namespace bioexplorer
