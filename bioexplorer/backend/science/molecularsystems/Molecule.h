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

#include <science/common/SDFGeometries.h>

namespace bioexplorer
{
namespace molecularsystems
{
const std::string KEY_UNDEFINED = "Undefined";
const std::string KEY_ATOM = "ATOM";
const std::string KEY_HETATM = "HETATM";
const std::string KEY_HEADER = "HEADER";
const std::string KEY_TITLE = "TITLE";
const std::string KEY_CONECT = "CONECT";
const std::string KEY_SEQRES = "SEQRES";
const std::string KEY_REMARK = "REMARK";

// Amino acids
static AminoAcidMap aminoAcidMap = {{".", {".", "."}},
                                    {"ALA", {"Alanine", "A"}},
                                    {"CYS", {"Cysteine", "C"}},
                                    {"ASP", {"Aspartic acid", "D"}},
                                    {"GLU", {"Glutamic acid", "E"}},
                                    {"PHE", {"Phenylalanine", "F"}},
                                    {"GLY", {"Glycine", "G"}},
                                    {"HIS", {"Histidine", "H"}},
                                    {"ILE", {"Isoleucine", "I"}},
                                    {"LYS", {"Lysine", "K"}},
                                    {"LEU", {"Leucine", "L"}},
                                    {"MET", {"Methionine", "M"}},
                                    {"ASN", {"Asparagine", "N"}},
                                    {"HYP", {"Hydroxyproline", "O"}},
                                    {"PRO", {"Proline", "P"}},
                                    {"GLN", {"Glutamine", "Q"}},
                                    {"ARG", {"Arginine", "R"}},
                                    {"SER", {"Serine", "S"}},
                                    {"THR", {"Threonine", "T"}},
                                    {"GLP", {"Pyroglutamatic", "U"}},
                                    {"VAL", {"Valine", "V"}},
                                    {"TRP", {"Tryptophan", "W"}},
                                    {"TYR", {"Tyrosine", "Y"}}};

// Protein color maps
typedef struct
{
    short r, g, b;
} RGBColorDetails;
using RGBColorDetailsMap = std::map<std::string, RGBColorDetails>;

static RGBColorDetailsMap atomColorMap = {
    {"H", {0xDF, 0xDF, 0xDF}},        {"He", {0xD9, 0xFF, 0xFF}},   {"Li", {0xCC, 0x80, 0xFF}},
    {"Be", {0xC2, 0xFF, 0x00}},       {"B", {0xFF, 0xB5, 0xB5}},    {"C", {0x90, 0x90, 0x90}},
    {"N", {0x30, 0x50, 0xF8}},        {"O", {0xFF, 0x0D, 0x0D}},    {"F", {0x9E, 0x05, 0x1}},
    {"Ne", {0xB3, 0xE3, 0xF5}},       {"Na", {0xAB, 0x5C, 0xF2}},   {"Mg", {0x8A, 0xFF, 0x00}},
    {"Al", {0xBF, 0xA6, 0xA6}},       {"Si", {0xF0, 0xC8, 0xA0}},   {"P", {0xFF, 0x80, 0x00}},
    {"S", {0xFF, 0xFF, 0x30}},        {"Cl", {0x1F, 0xF0, 0x1F}},   {"Ar", {0x80, 0xD1, 0xE3}},
    {"K", {0x8F, 0x40, 0xD4}},        {"Ca", {0x3D, 0xFF, 0x00}},   {"Sc", {0xE6, 0xE6, 0xE6}},
    {"Ti", {0xBF, 0xC2, 0xC7}},       {"V", {0xA6, 0xA6, 0xAB}},    {"Cr", {0x8A, 0x99, 0xC7}},
    {"Mn", {0x9C, 0x7A, 0xC7}},       {"Fe", {0xE0, 0x66, 0x33}},   {"Co", {0xF0, 0x90, 0xA0}},
    {"Ni", {0x50, 0xD0, 0x50}},       {"Cu", {0xC8, 0x80, 0x33}},   {"Zn", {0x7D, 0x80, 0xB0}},
    {"Ga", {0xC2, 0x8F, 0x8F}},       {"Ge", {0x66, 0x8F, 0x8F}},   {"As", {0xBD, 0x80, 0xE3}},
    {"Se", {0xFF, 0xA1, 0x00}},       {"Br", {0xA6, 0x29, 0x29}},   {"Kr", {0x5C, 0xB8, 0xD1}},
    {"Rb", {0x70, 0x2E, 0xB0}},       {"Sr", {0x00, 0xFF, 0x00}},   {"Y", {0x94, 0xFF, 0xFF}},
    {"Zr", {0x94, 0xE0, 0xE0}},       {"Nb", {0x73, 0xC2, 0xC9}},   {"Mo", {0x54, 0xB5, 0xB5}},
    {"Tc", {0x3B, 0x9E, 0x9E}},       {"Ru", {0x24, 0x8F, 0x8F}},   {"Rh", {0x0A, 0x7D, 0x8C}},
    {"Pd", {0x69, 0x85, 0x00}},       {"Ag", {0xC0, 0xC0, 0xC0}},   {"Cd", {0xFF, 0xD9, 0x8F}},
    {"In", {0xA6, 0x75, 0x73}},       {"Sn", {0x66, 0x80, 0x80}},   {"Sb", {0x9E, 0x63, 0xB5}},
    {"Te", {0xD4, 0x7A, 0x00}},       {"I", {0x94, 0x00, 0x94}},    {"Xe", {0x42, 0x9E, 0xB0}},
    {"Cs", {0x57, 0x17, 0x8F}},       {"Ba", {0x00, 0xC9, 0x00}},   {"La", {0x70, 0xD4, 0xFF}},
    {"Ce", {0xFF, 0xFF, 0xC7}},       {"Pr", {0xD9, 0xFF, 0xC7}},   {"Nd", {0xC7, 0xFF, 0xC7}},
    {"Pm", {0xA3, 0xFF, 0xC7}},       {"Sm", {0x8F, 0xFF, 0xC7}},   {"Eu", {0x61, 0xFF, 0xC7}},
    {"Gd", {0x45, 0xFF, 0xC7}},       {"Tb", {0x30, 0xFF, 0xC7}},   {"Dy", {0x1F, 0xFF, 0xC7}},
    {"Ho", {0x00, 0xFF, 0x9C}},       {"Er", {0x00, 0xE6, 0x75}},   {"Tm", {0x00, 0xD4, 0x52}},
    {"Yb", {0x00, 0xBF, 0x38}},       {"Lu", {0x00, 0xAB, 0x24}},   {"Hf", {0x4D, 0xC2, 0xFF}},
    {"Ta", {0x4D, 0xA6, 0xFF}},       {"W", {0x21, 0x94, 0xD6}},    {"Re", {0x26, 0x7D, 0xAB}},
    {"Os", {0x26, 0x66, 0x96}},       {"Ir", {0x17, 0x54, 0x87}},   {"Pt", {0xD0, 0xD0, 0xE0}},
    {"Au", {0xFF, 0xD1, 0x23}},       {"Hg", {0xB8, 0xB8, 0xD0}},   {"Tl", {0xA6, 0x54, 0x4D}},
    {"Pb", {0x57, 0x59, 0x61}},       {"Bi", {0x9E, 0x4F, 0xB5}},   {"Po", {0xAB, 0x5C, 0x00}},
    {"At", {0x75, 0x4F, 0x45}},       {"Rn", {0x42, 0x82, 0x96}},   {"Fr", {0x42, 0x00, 0x66}},
    {"Ra", {0x00, 0x7D, 0x00}},       {"Ac", {0x70, 0xAB, 0xFA}},   {"Th", {0x00, 0xBA, 0xFF}},
    {"Pa", {0x00, 0xA1, 0xFF}},       {"U", {0x00, 0x8F, 0xFF}},    {"Np", {0x00, 0x80, 0xFF}},
    {"Pu", {0x00, 0x6B, 0xFF}},       {"Am", {0x54, 0x5C, 0xF2}},   {"Cm", {0x78, 0x5C, 0xE3}},
    {"Bk", {0x8A, 0x4F, 0xE3}},       {"Cf", {0xA1, 0x36, 0xD4}},   {"Es", {0xB3, 0x1F, 0xD4}},
    {"Fm", {0xB3, 0x1F, 0xBA}},       {"Md", {0xB3, 0x0D, 0xA6}},   {"No", {0xBD, 0x0D, 0x87}},
    {"Lr", {0xC7, 0x00, 0x66}},       {"Rf", {0xCC, 0x00, 0x59}},   {"Db", {0xD1, 0x00, 0x4F}},
    {"Sg", {0xD9, 0x00, 0x45}},       {"Bh", {0xE0, 0x00, 0x38}},   {"Hs", {0xE6, 0x00, 0x2E}},
    {"Mt", {0xEB, 0x00, 0x26}},       {"none", {0xFF, 0xFF, 0xFF}}, {"O1", {0xFF, 0x0D, 0x0D}},
    {"selection", {0xFF, 0x00, 0x00}}};

/**
 * @brief The Molecule class implements the 3D representation of a molecule. The object also contains metadata attached
 * to the molecule itself, such as the amino acids sequence, or the chain ids for example. The current implementation
 * only supports PDB as an input format for the molecule data and metadata
 */
class Molecule : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new Molecule object
     *
     * @param scene The 3D scene where the glycans are added
     * @param chainIds IDs of chains to be used to construct the molecule object
     */
    Molecule(core::Scene& scene, const core::Vector3d& position, const core::Quaterniond& rotation,
             const size_ts& chainIds);

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
    const ResidueSequenceMap& getResidueSequences() const { return _residueSequenceMap; }

    /**
     * @brief Get the Sequences As String object
     *
     * @return StringMap
     */
    const StringMap getSequencesAsString() const;

protected:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _setAtomColorScheme();
    void _setChainColorScheme(const Palette& palette);
    void _setResiduesColorScheme(const Palette& palette);
    void _setAminoAcidSequenceColorScheme(const Palette& palette);
    void _setMaterialDiffuseColor(const size_t atomIndex, const RGBColorDetails& color);
    void _setMaterialDiffuseColor(const size_t atomIndex, const Color& color);

    // Geometry
    void _buildModel(const std::string& assemblyName, const std::string& name, const std::string& title,
                     const std::string& header, const ProteinRepresentation& representation,
                     const double atomRadiusMultiplier, const bool loadBonds);

    void _buildAtomicStruture(const ProteinRepresentation representation, const double atomRadiusMultiplier,
                              const bool surface, const bool loadBonds, common::ThreadSafeContainer& container);
    void _computeReqSetOffset();

    // IO
    void _readAtom(const std::string& line, const bool loadHydrogen);
    void _readSequence(const std::string& line);
    std::string _readHeader(const std::string& line);
    std::string _readTitle(const std::string& line);
    void _readRemark(const std::string& line);
    void _readConnect(const std::string& line);
    bool _loadChain(const size_t chainId);
    void _rescaleMesh(core::Model& model, const core::Vector3f& scale = {1.f, 1.f, 1.f});

    core::Scene& _scene;
    AtomMap _atomMap;
    Residues _residues;
    ResidueSequenceMap _residueSequenceMap;
    BondsMap _bondsMap;
    size_ts _chainIds;

    core::Vector2ui _aminoAcidRange;

    std::string _selectedAminoAcidSequence;
    Vector2uis _selectedAminoAcidRanges;
};
} // namespace molecularsystems
} // namespace bioexplorer
