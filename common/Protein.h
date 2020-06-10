/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_PROTEIN_H
#define BIOEXPLORER_PROTEIN_H

#include <api/BioExplorerParams.h>

#include <common/Node.h>
#include <common/types.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
class Protein : public Node
{
public:
    Protein(Scene& scene, const ProteinDescriptor& descriptor);

    // Color schemes
    void setColorScheme(const ColorScheme& colorScheme, const Palette& palette,
                        const size_ts& chainIds);

    // Amino acid sequence
    StringMap getSequencesAsString() const;
    void setAminoAcidSequenceAsString(const std::string& aminoAcidSequence)
    {
        _aminoAcidSequence = aminoAcidSequence;
        _aminoAcidRange = {0, 0};
    }
    void setAminoAcidSequenceAsRange(const Vector2ui& range)
    {
        _aminoAcidSequence = "";
        _aminoAcidRange = range;
    }
    const std::string& getAminoAcidSequence() const
    {
        return _aminoAcidSequence;
    }

    // Class member accessors
    AtomMap& getAtoms() { return _atomMap; }
    void setAtoms(const AtomMap& atoms) { _atomMap = atoms; }
    Residues& getResidues() { return _residues; }
    SequenceMap& getSequences() { return _sequenceMap; }
    const ProteinDescriptor& getDescriptor() const { return _descritpor; }

    void getGlycosilationSites(std::vector<Vector3f>& positions,
                               std::vector<Quaterniond>& rotations,
                               const std::vector<size_t>& siteIndices) const;

    void getGlucoseBindingSites(std::vector<Vector3f>& positions,
                                std::vector<Quaterniond>& rotations,
                                const std::vector<size_t>& siteIndices) const;

    std::map<std::string, size_ts> getGlycosylationSites(
        const size_ts& siteIndices) const;

private:
    // Analysis
    void _getSitesTransformations(
        std::vector<Vector3f>& positions, std::vector<Quaterniond>& rotations,
        const std::map<std::string, size_ts>& sites) const;

    // Color schemes
    void _setAtomColorScheme();
    void _setChainColorScheme(const Palette& palette);
    void _setRegionColorScheme(const Palette& palette, const size_ts& chainIds);
    void _setResiduesColorScheme(const Palette& palette);
    void _setAminoAcidSequenceColorScheme(const Palette& palette);
    void _setGlycosylationSiteColorScheme(const Palette& palette);

    void _setMaterialDiffuseColor(const size_t atomIndex,
                                  const RGBColor& color);
    void _setMaterialDiffuseColor(const size_t atomIndex, const Color& color);

    // IO
    void _readAtom(const std::string& line);
    void _readSequence(const std::string& line);
    void _readTitle(const std::string& line);
    void _readRemark(const std::string& line);
    void _readConnect(const std::string& line);
    bool _loadChain(const size_t chainId);
    void _buildModel(Model& model, const ProteinDescriptor& descriptor);

    // Class members
    ProteinDescriptor _descritpor;
    AtomMap _atomMap;
    Residues _residues;
    SequenceMap _sequenceMap;
    BondsMap _bondsMap;
    size_ts _chainIds;

    std::string _aminoAcidSequence;
    Vector2ui _aminoAcidRange;
    std::string _title;
    Boxf _bounds;
};
} // namespace bioexplorer

#endif // BIOEXPLORER_PROTEIN_H
