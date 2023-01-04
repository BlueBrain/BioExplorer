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

#include <plugin/common/SDFGeometries.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace brayns;
using namespace common;
using namespace details;

const std::string KEY_UNDEFINED = "Undefined";
const std::string KEY_ATOM = "ATOM";
const std::string KEY_HETATM = "HETATM";
const std::string KEY_HEADER = "HEADER";
const std::string KEY_TITLE = "TITLE";
const std::string KEY_CONECT = "CONECT";
const std::string KEY_SEQRES = "SEQRES";
const std::string KEY_REMARK = "REMARK";

/**
 * @brief The Molecule class implements the 3D representation of a molecule. The
 * object also contains metadata attached to the molecule itself, such as the
 * amino acids sequence, or the chain ids for example. The current
 * implementation only supports PDB as an input format for the molecule data and
 * metadata
 */
class Molecule : public SDFGeometries
{
public:
    /**
     * @brief Construct a new Molecule object
     *
     * @param scene The 3D scene where the glycans are added
     * @param chainIds IDs of chains to be used to construct the molecule object
     */
    Molecule(Scene& scene, const size_ts& chainIds);

    /**
     * @brief Get the Atoms object
     *
     * @return AtomMap& The map of atoms composing the molecule. The key of the
     * map is the id of the atom, as defined in the PDB file
     */
    const AtomMap& getAtoms() const { return _atomMap; }

    /**
     * @brief Get the Residues object
     *
     * @return Residues& The list of residues composing the molecule
     */
    const Residues& getResidues() const { return _residues; }

    /**
     * @brief Get the Sequences object
     *
     * @return SequenceMap& The map of acid amino sequences composing the
     * molecule. The key of the map is the id of the chain, as defined in the
     * PDB file
     */
    const ResidueSequenceMap& getResidueSequences() const
    {
        return _residueSequenceMap;
    }

    /**
     * @brief Get the Sequences As String object
     *
     * @return StringMap
     */
    const StringMap getSequencesAsString() const;

    /**
     * @brief Get the bounds of the molecule atomic structure
     *
     * @return const Boxf& Bounds of the molecule atomic structure
     */
    const Boxf& getBounds() const { return _bounds; };

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
                     const double atomRadiusMultiplier, const bool loadBonds);

    void _buildAtomicStruture(const ProteinRepresentation representation,
                              const double atomRadiusMultiplier,
                              const bool surface, const bool loadBonds,
                              ThreadSafeContainer& container);
    void _computeReqSetOffset();

    // IO
    void _readAtom(const std::string& line, const bool loadHydrogen);
    void _readSequence(const std::string& line);
    std::string _readHeader(const std::string& line);
    std::string _readTitle(const std::string& line);
    void _readRemark(const std::string& line);
    void _readConnect(const std::string& line);
    bool _loadChain(const size_t chainId);
    void _rescaleMesh(Model& model, const Vector3f& scale = {1.f, 1.f, 1.f});

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
} // namespace molecularsystems
} // namespace bioexplorer
