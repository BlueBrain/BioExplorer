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

#include "AdvancedProteinLoader.h"

#include "../log.h"

#include <brayns/common/utils/utils.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

#include <assert.h>
#include <fstream>

namespace
{
const auto LOADER_NAME = "Advanced protein loader";

const strings LOADER_EXTENSIONS{"pdb", "pdb1", "ent"};
} // namespace

namespace brayns
{
template <>
inline std::vector<std::pair<std::string, ColorScheme>> enumMap()
{
    return {{"none", ColorScheme::none},
            {"atoms", ColorScheme::atoms},
            {"chains", ColorScheme::chains},
            {"residues", ColorScheme::residues},
            {"transmembrane_sequence", ColorScheme::transmembrane_sequence},
            {"glycosylation_site", ColorScheme::glycosylation_site}};
}
} // namespace brayns

/** Structure defining an atom radius in microns
 */
struct AtomicRadius
{
    std::string Symbol;
    float radius;
    int index;
};
const float DEFAULT_RADIUS = 25.f;

/** Structure defining the color of atoms according to the JMol Scheme
 */
struct ProteinColorMap
{
    std::string symbol;
    short R, G, B;
};

// Protein color maps
const size_t PROTEIN_COLORMAP_SIZE = 113;
static ProteinColorMap colorMap[PROTEIN_COLORMAP_SIZE] = {
    {"H", 0xDF, 0xDF, 0xDF},
    {"He", 0xD9, 0xFF, 0xFF},
    {"Li", 0xCC, 0x80, 0xFF},
    {"Be", 0xC2, 0xFF, 0x00},
    {"B", 0xFF, 0xB5, 0xB5},
    {"C", 0x90, 0x90, 0x90},
    {"N", 0x30, 0x50, 0xF8},
    {"O", 0xFF, 0x0D, 0x0D},
    {"F", 0x9E, 0x05, 0x1},
    {"Ne", 0xB3, 0xE3, 0xF5},
    {"Na", 0xAB, 0x5C, 0xF2},
    {"Mg", 0x8A, 0xFF, 0x00},
    {"Al", 0xBF, 0xA6, 0xA6},
    {"Si", 0xF0, 0xC8, 0xA0},
    {"P", 0xFF, 0x80, 0x00},
    {"S", 0xFF, 0xFF, 0x30},
    {"Cl", 0x1F, 0xF0, 0x1F},
    {"Ar", 0x80, 0xD1, 0xE3},
    {"K", 0x8F, 0x40, 0xD4},
    {"Ca", 0x3D, 0xFF, 0x00},
    {"Sc", 0xE6, 0xE6, 0xE6},
    {"Ti", 0xBF, 0xC2, 0xC7},
    {"V", 0xA6, 0xA6, 0xAB},
    {"Cr", 0x8A, 0x99, 0xC7},
    {"Mn", 0x9C, 0x7A, 0xC7},
    {"Fe", 0xE0, 0x66, 0x33},
    {"Co", 0xF0, 0x90, 0xA0},
    {"Ni", 0x50, 0xD0, 0x50},
    {"Cu", 0xC8, 0x80, 0x33},
    {"Zn", 0x7D, 0x80, 0xB0},
    {"Ga", 0xC2, 0x8F, 0x8F},
    {"Ge", 0x66, 0x8F, 0x8F},
    {"As", 0xBD, 0x80, 0xE3},
    {"Se", 0xFF, 0xA1, 0x00},
    {"Br", 0xA6, 0x29, 0x29},
    {"Kr", 0x5C, 0xB8, 0xD1},
    {"Rb", 0x70, 0x2E, 0xB0},
    {"Sr", 0x00, 0xFF, 0x00},
    {"Y", 0x94, 0xFF, 0xFF},
    {"Zr", 0x94, 0xE0, 0xE0},
    {"Nb", 0x73, 0xC2, 0xC9},
    {"Mo", 0x54, 0xB5, 0xB5},
    {"Tc", 0x3B, 0x9E, 0x9E},
    {"Ru", 0x24, 0x8F, 0x8F},
    {"Rh", 0x0A, 0x7D, 0x8C},
    {"Pd", 0x69, 0x85, 0x00},
    {"Ag", 0xC0, 0xC0, 0xC0},
    {"Cd", 0xFF, 0xD9, 0x8F},
    {"In", 0xA6, 0x75, 0x73},
    {"Sn", 0x66, 0x80, 0x80},
    {"Sb", 0x9E, 0x63, 0xB5},
    {"Te", 0xD4, 0x7A, 0x00},
    {"I", 0x94, 0x00, 0x94},
    {"Xe", 0x42, 0x9E, 0xB0},
    {"Cs", 0x57, 0x17, 0x8F},
    {"Ba", 0x00, 0xC9, 0x00},
    {"La", 0x70, 0xD4, 0xFF},
    {"Ce", 0xFF, 0xFF, 0xC7},
    {"Pr", 0xD9, 0xFF, 0xC7},
    {"Nd", 0xC7, 0xFF, 0xC7},
    {"Pm", 0xA3, 0xFF, 0xC7},
    {"Sm", 0x8F, 0xFF, 0xC7},
    {"Eu", 0x61, 0xFF, 0xC7},
    {"Gd", 0x45, 0xFF, 0xC7},
    {"Tb", 0x30, 0xFF, 0xC7},
    {"Dy", 0x1F, 0xFF, 0xC7},
    {"Ho", 0x00, 0xFF, 0x9C},
    {"Er", 0x00, 0xE6, 0x75},
    {"Tm", 0x00, 0xD4, 0x52},
    {"Yb", 0x00, 0xBF, 0x38},
    {"Lu", 0x00, 0xAB, 0x24},
    {"Hf", 0x4D, 0xC2, 0xFF},
    {"Ta", 0x4D, 0xA6, 0xFF},
    {"W", 0x21, 0x94, 0xD6},
    {"Re", 0x26, 0x7D, 0xAB},
    {"Os", 0x26, 0x66, 0x96},
    {"Ir", 0x17, 0x54, 0x87},
    {"Pt", 0xD0, 0xD0, 0xE0},
    {"Au", 0xFF, 0xD1, 0x23},
    {"Hg", 0xB8, 0xB8, 0xD0},
    {"Tl", 0xA6, 0x54, 0x4D},
    {"Pb", 0x57, 0x59, 0x61},
    {"Bi", 0x9E, 0x4F, 0xB5},
    {"Po", 0xAB, 0x5C, 0x00},
    {"At", 0x75, 0x4F, 0x45},
    {"Rn", 0x42, 0x82, 0x96},
    {"Fr", 0x42, 0x00, 0x66},
    {"Ra", 0x00, 0x7D, 0x00},
    {"Ac", 0x70, 0xAB, 0xFA},
    {"Th", 0x00, 0xBA, 0xFF},
    {"Pa", 0x00, 0xA1, 0xFF},
    {"U", 0x00, 0x8F, 0xFF},
    {"Np", 0x00, 0x80, 0xFF},
    {"Pu", 0x00, 0x6B, 0xFF},
    {"Am", 0x54, 0x5C, 0xF2},
    {"Cm", 0x78, 0x5C, 0xE3},
    {"Bk", 0x8A, 0x4F, 0xE3},
    {"Cf", 0xA1, 0x36, 0xD4},
    {"Es", 0xB3, 0x1F, 0xD4},
    {"Fm", 0xB3, 0x1F, 0xBA},
    {"Md", 0xB3, 0x0D, 0xA6},
    {"No", 0xBD, 0x0D, 0x87},
    {"Lr", 0xC7, 0x00, 0x66},
    {"Rf", 0xCC, 0x00, 0x59},
    {"Db", 0xD1, 0x00, 0x4F},
    {"Sg", 0xD9, 0x00, 0x45},
    {"Bh", 0xE0, 0x00, 0x38},
    {"Hs", 0xE6, 0x00, 0x2E},
    {"Mt", 0xEB, 0x00, 0x26},

    // TODO
    {"", 0xFF, 0xFF, 0xFF},
    {"", 0xFF, 0xFF, 0xFF},
    {"", 0xFF, 0xFF, 0xFF},
    {"", 0xFF, 0xFF, 0xFF}};

// Atomic radii in microns
static AtomicRadius atomic_radii[PROTEIN_COLORMAP_SIZE] = {{"C", 67.f, 1},
                                                           {"N", 56.f, 2},
                                                           {"O", 48.f, 3},
                                                           {"H", 53.f, 4},
                                                           {"B", 87.f, 5},
                                                           {"F", 42.f, 6},
                                                           {"P", 98.f, 7},
                                                           {"S", 88.f, 8},
                                                           {"V", 171.f, 9},
                                                           {"K", 243.f, 10},
                                                           {"HE", 31.f, 11},
                                                           {"LI", 167.f, 12},
                                                           {"BE", 112.f, 13},
                                                           {"NE", 38.f, 14},
                                                           {"NA", 190.f, 15},
                                                           {"MG", 145.f, 16},
                                                           {"AL", 118.f, 17},
                                                           {"SI", 111.f, 18},
                                                           {"CL", 79.f, 19},
                                                           {"AR", 71.f, 20},
                                                           {"CA", 194.f, 21},
                                                           {"SC", 184.f, 22},
                                                           {"TI", 176.f, 23},
                                                           {"CR", 166.f, 24},
                                                           {"MN", 161.f, 25},
                                                           {"FE", 156.f, 26},
                                                           {"CO", 152.f, 27},
                                                           {"NI", 149.f, 28},
                                                           {"CU", 145.f, 29},
                                                           {"ZN", 142.f, 30},
                                                           {"GA", 136.f, 31},
                                                           {"GE", 125.f, 32},
                                                           {"AS", 114.f, 33},
                                                           {"SE", 103.f, 34},
                                                           {"BR", 94.f, 35},
                                                           {"KR", 88.f, 36},

                                                           // TODO
                                                           {"OD1", 25.f, 37},
                                                           {"OD2", 25.f, 38},
                                                           {"CG1", 25.f, 39},
                                                           {"CG2", 25.f, 40},
                                                           {"CD1", 25.f, 41},
                                                           {"CB", 25.f, 42},
                                                           {"CG", 25.f, 43},
                                                           {"CD", 25.f, 44},
                                                           {"OE1", 25.f, 45},
                                                           {"NE2", 25.f, 46},
                                                           {"CZ", 25.f, 47},
                                                           {"NH1", 25.f, 48},
                                                           {"NH2", 25.f, 49},
                                                           {"CD2", 25.f, 50},
                                                           {"CE1", 25.f, 51},
                                                           {"CE2", 25.f, 52},
                                                           {"CE", 25.f, 53},
                                                           {"NZ", 25.f, 54},
                                                           {"OH", 25.f, 55},
                                                           {"CE", 25.f, 56},
                                                           {"ND1", 25.f, 57},
                                                           {"ND2", 25.f, 58},
                                                           {"OXT", 25.f, 59},
                                                           {"OG1", 25.f, 60},
                                                           {"NE1", 25.f, 61},
                                                           {"CE3", 25.f, 62},
                                                           {"CZ2", 25.f, 63},
                                                           {"CZ3", 25.f, 64},
                                                           {"CH2", 25.f, 65},
                                                           {"OE2", 25.f, 66},
                                                           {"OG", 25.f, 67},
                                                           {"OE2", 25.f, 68},
                                                           {"SD", 25.f, 69},
                                                           {"SG", 25.f, 70},
                                                           {"C1*", 25.f, 71},
                                                           {"C2", 25.f, 72},
                                                           {"C2*", 25.f, 73},
                                                           {"C3*", 25.f, 74},
                                                           {"C4", 25.f, 75},
                                                           {"C4*", 25.f, 76},
                                                           {"C5", 25.f, 77},
                                                           {"C5*", 25.f, 78},
                                                           {"C5M", 25.f, 79},
                                                           {"C6", 25.f, 80},
                                                           {"C8", 25.f, 81},
                                                           {"H1", 25.f, 82},
                                                           {"H1*", 25.f, 83},
                                                           {"H2", 25.f, 84},
                                                           {"H2*", 25.f, 85},
                                                           {"H3", 25.f, 86},
                                                           {"H3*", 25.f, 87},
                                                           {"H3P", 25.f, 88},
                                                           {"H4", 25.f, 89},
                                                           {"H4*", 25.f, 90},
                                                           {"H5", 25.f, 91},
                                                           {"H5*", 25.f, 92},
                                                           {"H5M", 25.f, 93},
                                                           {"H6", 25.f, 94},
                                                           {"H8", 25.f, 95},
                                                           {"N1", 25.f, 96},
                                                           {"N2", 25.f, 97},
                                                           {"N3", 25.f, 98},
                                                           {"N4", 25.f, 99},
                                                           {"N6", 25.f, 100},
                                                           {"N7", 25.f, 101},
                                                           {"N9", 25.f, 102},
                                                           {"O1P", 25.f, 103},
                                                           {"O2", 25.f, 104},
                                                           {"O2P", 25.f, 105},
                                                           {"O3*", 25.f, 106},
                                                           {"O3P", 25.f, 107},
                                                           {"O4", 25.f, 108},
                                                           {"O4*", 25.f, 109},
                                                           {"O5*", 25.f, 110},
                                                           {"O6", 25.f, 111},
                                                           {"OXT", 25.f, 112},
                                                           {"P", 25.f, 113}};

// Amino acids
static AminoAcidMap aminoAcidMap = {{"ALA", {"Alanine", "A"}},
                                    {"ARG", {"Arginine", "R"}},
                                    {"ASN", {"Asparagine", "N"}},
                                    {"ASP", {"Aspartic acid", "D"}},
                                    {"CYS", {"Cysteine", "C"}},
                                    {"GLN", {"Glutamine", "Q"}},
                                    {"GLU", {"Glutamic acid", "E"}},
                                    {"GLY", {"Glycine", "G"}},
                                    {"HIS", {"Histidine", "H"}},
                                    {"ILE", {"Isoleucine", "I"}},
                                    {"LEU", {"Leucine", "L"}},
                                    {"LYS", {"Lysine", "K"}},
                                    {"MET", {"Methionine", "M"}},
                                    {"PHE", {"Phenylalanine", "F"}},
                                    {"SER", {"Proline Pro P Serine", "S"}},
                                    {"THR", {"Threonine", "T"}},
                                    {"TRP", {"Tryptophan", "W"}},
                                    {"TYR", {"Tyrosine", "Y"}},
                                    {"VAL", {"Valine", "V"}}};

std::string& ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::ptr_fun<int, int>(std::isgraph)));
    return s;
}

std::string& rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::ptr_fun<int, int>(std::isgraph))
                .base(),
            s.end());
    return s;
}

std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

AdvancedProteinLoader::AdvancedProteinLoader(
    brayns::Scene& scene, const brayns::PropertyMap& properties)
    : Loader(scene)
    , _defaults(properties)
{
}

AdvancedProteinLoader::AdvancedProteinLoader(
    brayns::Scene& scene, const brayns::GeometryParameters& params)
    : Loader(scene)
{
    _defaults.setProperty(PROP_RADIUS_MULTIPLIER);
    _defaults.setProperty(PROP_PROTEIN_COLOR_SCHEME);
    _defaults.setProperty(PROP_TRANSMEMBRANE_SEQUENCE);
}

void AdvancedProteinLoader::readAtom(const std::string& line,
                                     const ColorScheme colorScheme,
                                     const float radiusMultiplier, Atoms& atoms,
                                     Residues& residues) const
{
    Atom atom;
    // --------------------------------------------------------------------
    // COLUMNS DATA TYPE    FIELD     DEFINITION
    // --------------------------------------------------------------------
    // 1 - 6   Record name  "ATOM "
    // 7 - 11  Integer      serial     Atom serial number
    // 13 - 16 Atom         name       Atom name
    // 17      Character    altLoc     Alternate location indicator
    // 18 - 20 Residue name resName    Residue name
    // 22      Character    chainID    Chain identifier
    // 23 - 26 Integer      resSeq     Residue sequence number
    // 27      AChar        iCode      Code for insertion of residues
    // 31 - 38 Real(8.3)    x          Orthogonal coords for X in angstroms
    // 39 - 46 Real(8.3)    y          Orthogonal coords for Y in Angstroms
    // 47 - 54 Real(8.3)    z          Orthogonal coords for Z in Angstroms
    // 55 - 60 Real(6.2)    occupancy  Occupancy
    // 61 - 66 Real(6.2)    tempFactor Temperature factor
    // 77 - 78 LString(2)   element    Element symbol, right-justified
    // 79 - 80 LString(2)   charge     Charge on the atom
    // --------------------------------------------------------------------

    atom.serial = static_cast<size_t>(atoi(line.substr(6, 4).c_str()));
    std::string s = line.substr(12, 4);
    atom.name = trim(s);

    s = line.substr(16, 1);
    atom.altLoc = trim(s);

    s = line.substr(17, 3);
    atom.resName = trim(s);
    residues.insert(atom.resName);

    s = line.substr(21, 1);
    atom.chainId = trim(s);

    atom.reqSeq = static_cast<size_t>(atoi(line.substr(22, 4).c_str()));

    atom.iCode = line.substr(26, 1);

    atom.position.x = static_cast<float>(atof(line.substr(30, 8).c_str()));
    atom.position.y = static_cast<float>(atof(line.substr(38, 8).c_str()));
    atom.position.z = static_cast<float>(atof(line.substr(46, 8).c_str()));

    atom.occupancy = static_cast<float>(atof(line.substr(54, 6).c_str()));

    atom.tempFactor = static_cast<float>(atof(line.substr(60, 6).c_str()));

    s = line.substr(76, 2);
    atom.element = trim(s);

    s = line.substr(78, 2);
    atom.charge = trim(s);

    // Material Id
    atom.materialId = 0;
    size_t i = 0;
    bool found = false;
    while (!found && i < PROTEIN_COLORMAP_SIZE)
    {
        if (atom.element == colorMap[i].symbol)
        {
            found = true;
            switch (colorScheme)
            {
            case ColorScheme::chains:
                atom.materialId = static_cast<size_t>(atom.chainId[0]) - 64;
                break;
            case ColorScheme::residues:
                atom.materialId = static_cast<size_t>(
                    std::distance(residues.begin(),
                                  residues.find(atom.resName)));
                break;
            default:
                atom.materialId = i;
                break;
            }
        }
        ++i;
    }

    // Radius
    atom.radius = DEFAULT_RADIUS;
    i = 0;
    found = false;
    while (!found && i < PROTEIN_COLORMAP_SIZE)
    {
        if (atom.element == atomic_radii[i].Symbol)
        {
            atom.radius = atomic_radii[i].radius;
            found = true;
        }
        ++i;
    }

    // Convert position from nanometers
    atom.position = 0.01f * atom.position;

    // Convert radius from angstrom
    atom.radius = 0.0001f * atom.radius * static_cast<float>(radiusMultiplier);

    atoms.push_back(atom);
}

void AdvancedProteinLoader::readSequence(const std::string& line,
                                         SequenceMap& sequenceMap) const
{
    // -------------------------------------------------------------------------
    // COLUMNS TYPE      FIELD    DEFINITION
    // -------------------------------------------------------------------------
    // 1 - 6   Record name "SEQRES"
    // 8 - 10  Integer   serNum   Serial number of the SEQRES record for the
    //                            current chain. Starts at 1 and increments by
    //                            one each line. Reset to 1 for each chain.
    // 12      Character chainID  Chain identifier. This may be any single legal
    //                            character, including a blank which is is used
    //                            if there is only one chain
    // 14 - 17 Integer   numRes   Number of residues in the chain. This value is
    //                            repeated on every record.
    // 20 - 22 String    resName  Residue name
    // 24 - 26 ...
    // -------------------------------------------------------------------------

    std::string s = line.substr(11, 1);
    size_t chainId = static_cast<size_t>(s[0]) - 64;

    Sequence& sequence = sequenceMap[chainId];
    sequence.serNum = static_cast<size_t>(atoi(line.substr(7, 3).c_str()));
    sequence.numRes = static_cast<size_t>(atoi(line.substr(13, 4).c_str()));

    for (size_t i = 19; i < line.length(); i += 4)
    {
        s = line.substr(i, 4);
        s = trim(s);
        if (!s.empty())
            sequence.resNames.push_back(s);
    }
}

brayns::ModelDescriptorPtr AdvancedProteinLoader::importFromFile(
    const std::string& fileName, const brayns::LoaderProgress&,
    const brayns::PropertyMap& inProperties) const
{
    Atoms atoms;
    Residues residues;
    SequenceMap sequenceMap;

    // Fill property map since the actual property types are known now.
    brayns::PropertyMap properties = _defaults;
    properties.merge(inProperties);

    const double radiusMultiplier =
        properties.getProperty<double>(PROP_RADIUS_MULTIPLIER.name, 1.0);

    const auto colorScheme = brayns::stringToEnum<ColorScheme>(
        properties.getProperty<std::string>(PROP_PROTEIN_COLOR_SCHEME.name));

    const auto transmembraneSequence =
        properties.getProperty<std::string>(PROP_TRANSMEMBRANE_SEQUENCE.name);

    std::ifstream file(fileName.c_str());
    if (!file.is_open())
        throw std::runtime_error("Could not open " + fileName);

    size_t lineIndex{0};

    while (file.good())
    {
        std::string line;
        std::getline(file, line);
        if (line.find("ATOM") == 0 /* || line.find("HETATM") == 0*/)
        {
            readAtom(line, colorScheme, radiusMultiplier, atoms, residues);
        }
        if (line.find("SEQRES") == 0)
        {
            readSequence(line, sequenceMap);
        }
    }
    file.close();

    // Sequences
    PLUGIN_INFO << "--------------------------------------------------"
                << std::endl;
    PLUGIN_INFO << "Sequences" << std::endl;
    std::map<std::string, std::string> sequencesAsStrings;
    for (const auto& sequence : sequenceMap)
    {
        std::string s;
        for (const auto& resName : sequence.second.resNames)
        {
            s = s + aminoAcidMap[resName].shortName;
            if (s.length() % 60 == 0)
                s = s + "\n";
        }
        sequencesAsStrings[std::to_string(sequence.first)] = s;
        PLUGIN_INFO << sequence.first << " (" << sequence.second.resNames.size()
                    << "): " << s << std::endl;
    }
    PLUGIN_INFO << "--------------------------------------------------"
                << std::endl;

    auto model = _scene.createModel();

    // Location color scheme
    switch (colorScheme)
    {
    case ColorScheme::transmembrane_sequence:
        for (const auto& sequence : sequenceMap)
        {
            std::string shortSequence;
            for (const auto& resName : sequence.second.resNames)
                shortSequence += aminoAcidMap[resName].shortName;

            const auto sequencePosition =
                shortSequence.find(transmembraneSequence);
            if (sequencePosition != -1)
            {
                PLUGIN_INFO << transmembraneSequence
                            << " was found at position " << sequencePosition
                            << std::endl;
                size_t atomCount = 0;
                size_t minSeq = 1e6;
                size_t maxSeq = 0;
                for (auto& atom : atoms)
                {
                    minSeq = std::min(minSeq, atom.reqSeq);
                    maxSeq = std::max(maxSeq, atom.reqSeq);
                    if (atom.reqSeq >= sequencePosition &&
                        atom.reqSeq <
                            sequencePosition + transmembraneSequence.length())
                    {
                        atom.materialId = 1;
                        ++atomCount;
                    }
                    else
                        atom.materialId = 0;
                }
                PLUGIN_INFO << atomCount << "[" << minSeq << "," << maxSeq
                            << "] atoms where colored" << std::endl;
            }
            else
            {
                PLUGIN_ERROR << transmembraneSequence << " was not found in "
                             << shortSequence << std::endl;
            }
        }
        break;
    case ColorScheme::glycosylation_site:
        for (auto& atom : atoms)
        {
            if (atom.resName == "ASN")
                atom.materialId = 0;
            else
                atom.materialId = static_cast<size_t>(atom.chainId[0]) - 63;
        }
        break;
    }

    // Materials
    std::set<size_t> materialIds;
    for (const auto& atom : atoms)
    {
        const auto materialId = atom.materialId;
        materialIds.insert(materialId);
    }

    for (const auto& materialId : materialIds)
    {
        auto material =
            model->createMaterial(materialId, colorMap[materialId].symbol);
        material->setDiffuseColor({colorMap[materialId].R / 255.f,
                                   colorMap[materialId].G / 255.f,
                                   colorMap[materialId].B / 255.f});
    }

    // Build 3d models according to atoms
    for (const auto& atom : atoms)
        model->addSphere(atom.materialId, {atom.position, atom.radius});

    // Transformation
    brayns::Transformation transformation;
    transformation.setRotationCenter(model->getBounds().getCenter());

    // Metadata
    // Create model
    brayns::ModelMetadata metadata;
    for (const auto& sequence : sequencesAsStrings)
        metadata[sequence.first] = sequence.second;

    metadata["Transmembrane Sequence"] = transmembraneSequence;

    auto modelDescriptor =
        std::make_shared<brayns::ModelDescriptor>(std::move(model), fileName,
                                                  metadata);
    modelDescriptor->setTransformation(transformation);
    return modelDescriptor;
}

std::string AdvancedProteinLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> AdvancedProteinLoader::getSupportedExtensions() const
{
    return LOADER_EXTENSIONS;
}

bool AdvancedProteinLoader::isSupported(const std::string& filename,
                                        const std::string& extension) const
{
    const auto ends_with = [](const std::string& value,
                              const std::string& ending) {
        if (ending.size() > value.size())
            return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    };

    for (const auto& name : LOADER_EXTENSIONS)
        if (ends_with(filename, name))
            return true;

    const auto contains = [](const std::string& value,
                             const std::string& keyword) {
        if (value.size() < keyword.size())
            return false;

        const auto lastSlash = value.find_last_of("/");
        std::string compareTo = value;
        if (lastSlash != std::string::npos)
            compareTo = value.substr(lastSlash + 1);
        return compareTo.find(keyword) != std::string::npos;
    };

    return false;
}

brayns::PropertyMap AdvancedProteinLoader::getProperties() const
{
    return _defaults;
}
