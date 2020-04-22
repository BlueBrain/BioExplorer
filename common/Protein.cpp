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

#include "Protein.h"

#include <common/log.h>
#include <common/utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

#include <fstream>

// Atomic radii in microns
static AtomicRadii atomicRadii = {{{"C"}, {67.f}},
                                  {{"N"}, {56.f}},
                                  {{"O"}, {48.f}},
                                  {{"H"}, {53.f}},
                                  {{"B"}, {87.f}},
                                  {{"F"}, {42.f}},
                                  {{"P"}, {98.f}},
                                  {{"S"}, {88.f}},
                                  {{"V"}, {171.f}},
                                  {{"K"}, {243.f}},
                                  {{"HE"}, {31.f}},
                                  {{"LI"}, {167.f}},
                                  {{"BE"}, {112.f}},
                                  {{"NE"}, {38.f}},
                                  {{"NA"}, {190.f}},
                                  {{"MG"}, {145.f}},
                                  {{"AL"}, {118.f}},
                                  {{"SI"}, {111.f}},
                                  {{"CL"}, {79.f}},
                                  {{"AR"}, {71.f}},
                                  {{"CA"}, {194.f}},
                                  {{"SC"}, {184.f}},
                                  {{"TI"}, {176.f}},
                                  {{"CR"}, {166.f}},
                                  {{"MN"}, {161.f}},
                                  {{"FE"}, {156.f}},
                                  {{"CO"}, {152.f}},
                                  {{"NI"}, {149.f}},
                                  {{"CU"}, {145.f}},
                                  {{"ZN"}, {142.f}},
                                  {{"GA"}, {136.f}},
                                  {{"GE"}, {125.f}},
                                  {{"AS"}, {114.f}},
                                  {{"SE"}, {103.f}},
                                  {{"BR"}, {94.f}},
                                  {{"KR"}, {88.f}},
                                  // TODO
                                  {{"OD1"}, {25.f}},
                                  {{"OD2"}, {25.f}},
                                  {{"CG1"}, {25.f}},
                                  {{"CG2"}, {25.f}},
                                  {{"CD1"}, {25.f}},
                                  {{"CB"}, {25.f}},
                                  {{"CG"}, {25.f}},
                                  {{"CD"}, {25.f}},
                                  {{"OE1"}, {25.f}},
                                  {{"NE2"}, {25.f}},
                                  {{"CZ"}, {25.f}},
                                  {{"NH1"}, {25.f}},
                                  {{"NH2"}, {25.f}},
                                  {{"CD2"}, {25.f}},
                                  {{"CE1"}, {25.f}},
                                  {{"CE2"}, {25.f}},
                                  {{"CE"}, {25.f}},
                                  {{"NZ"}, {25.f}},
                                  {{"OH"}, {25.f}},
                                  {{"CE"}, {25.f}},
                                  {{"ND1"}, {25.f}},
                                  {{"ND2"}, {25.f}},
                                  {{"OXT"}, {25.f}},
                                  {{"OG1"}, {25.f}},
                                  {{"NE1"}, {25.f}},
                                  {{"CE3"}, {25.f}},
                                  {{"CZ2"}, {25.f}},
                                  {{"CZ3"}, {25.f}},
                                  {{"CH2"}, {25.f}},
                                  {{"OE2"}, {25.f}},
                                  {{"OG"}, {25.f}},
                                  {{"OE2"}, {25.f}},
                                  {{"SD"}, {25.f}},
                                  {{"SG"}, {25.f}},
                                  {{"C1*"}, {25.f}},
                                  {{"C2"}, {25.f}},
                                  {{"C2*"}, {25.f}},
                                  {{"C3*"}, {25.f}},
                                  {{"C4"}, {25.f}},
                                  {{"C4*"}, {25.f}},
                                  {{"C5"}, {25.f}},
                                  {{"C5*"}, {25.f}},
                                  {{"C5M"}, {25.f}},
                                  {{"C6"}, {25.f}},
                                  {{"C8"}, {25.f}},
                                  {{"H1"}, {25.f}},
                                  {{"H1*"}, {25.f}},
                                  {{"H2"}, {25.f}},
                                  {{"H2*"}, {25.f}},
                                  {{"H3"}, {25.f}},
                                  {{"H3*"}, {25.f}},
                                  {{"H3P"}, {25.f}},
                                  {{"H4"}, {25.f}},
                                  {{"H4*"}, {25.f}},
                                  {{"H5"}, {25.f}},
                                  {{"H5*"}, {25.f}},
                                  {{"H5M"}, {25.f}},
                                  {{"H6"}, {25.f}},
                                  {{"H8"}, {25.f}},
                                  {{"N1"}, {25.f}},
                                  {{"N2"}, {25.f}},
                                  {{"N3"}, {25.f}},
                                  {{"N4"}, {25.f}},
                                  {{"N6"}, {25.f}},
                                  {{"N7"}, {25.f}},
                                  {{"N9"}, {25.f}},
                                  {{"O1P"}, {25.f}},
                                  {{"O2"}, {25.f}},
                                  {{"O2P"}, {25.f}},
                                  {{"O3*"}, {25.f}},
                                  {{"O3P"}, {25.f}},
                                  {{"O4"}, {25.f}},
                                  {{"O4*"}, {25.f}},
                                  {{"O5*"}, {25.f}},
                                  {{"O6"}, {25.f}},
                                  {{"OXT"}, {25.f}},
                                  {{"P"}, 25.f}};

const float BOND_RADIUS = 0.0025f;
const float DEFAULT_STICK_DISTANCE = 0.016f;

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
static RGBColorMap atomColorMap = {
    {"H", {0xDF, 0xDF, 0xDF}},        {"He", {0xD9, 0xFF, 0xFF}},
    {"Li", {0xCC, 0x80, 0xFF}},       {"Be", {0xC2, 0xFF, 0x00}},
    {"B", {0xFF, 0xB5, 0xB5}},        {"C", {0x90, 0x90, 0x90}},
    {"N", {0x30, 0x50, 0xF8}},        {"O", {0xFF, 0x0D, 0x0D}},
    {"F", {0x9E, 0x05, 0x1}},         {"Ne", {0xB3, 0xE3, 0xF5}},
    {"Na", {0xAB, 0x5C, 0xF2}},       {"Mg", {0x8A, 0xFF, 0x00}},
    {"Al", {0xBF, 0xA6, 0xA6}},       {"Si", {0xF0, 0xC8, 0xA0}},
    {"P", {0xFF, 0x80, 0x00}},        {"S", {0xFF, 0xFF, 0x30}},
    {"Cl", {0x1F, 0xF0, 0x1F}},       {"Ar", {0x80, 0xD1, 0xE3}},
    {"K", {0x8F, 0x40, 0xD4}},        {"Ca", {0x3D, 0xFF, 0x00}},
    {"Sc", {0xE6, 0xE6, 0xE6}},       {"Ti", {0xBF, 0xC2, 0xC7}},
    {"V", {0xA6, 0xA6, 0xAB}},        {"Cr", {0x8A, 0x99, 0xC7}},
    {"Mn", {0x9C, 0x7A, 0xC7}},       {"Fe", {0xE0, 0x66, 0x33}},
    {"Co", {0xF0, 0x90, 0xA0}},       {"Ni", {0x50, 0xD0, 0x50}},
    {"Cu", {0xC8, 0x80, 0x33}},       {"Zn", {0x7D, 0x80, 0xB0}},
    {"Ga", {0xC2, 0x8F, 0x8F}},       {"Ge", {0x66, 0x8F, 0x8F}},
    {"As", {0xBD, 0x80, 0xE3}},       {"Se", {0xFF, 0xA1, 0x00}},
    {"Br", {0xA6, 0x29, 0x29}},       {"Kr", {0x5C, 0xB8, 0xD1}},
    {"Rb", {0x70, 0x2E, 0xB0}},       {"Sr", {0x00, 0xFF, 0x00}},
    {"Y", {0x94, 0xFF, 0xFF}},        {"Zr", {0x94, 0xE0, 0xE0}},
    {"Nb", {0x73, 0xC2, 0xC9}},       {"Mo", {0x54, 0xB5, 0xB5}},
    {"Tc", {0x3B, 0x9E, 0x9E}},       {"Ru", {0x24, 0x8F, 0x8F}},
    {"Rh", {0x0A, 0x7D, 0x8C}},       {"Pd", {0x69, 0x85, 0x00}},
    {"Ag", {0xC0, 0xC0, 0xC0}},       {"Cd", {0xFF, 0xD9, 0x8F}},
    {"In", {0xA6, 0x75, 0x73}},       {"Sn", {0x66, 0x80, 0x80}},
    {"Sb", {0x9E, 0x63, 0xB5}},       {"Te", {0xD4, 0x7A, 0x00}},
    {"I", {0x94, 0x00, 0x94}},        {"Xe", {0x42, 0x9E, 0xB0}},
    {"Cs", {0x57, 0x17, 0x8F}},       {"Ba", {0x00, 0xC9, 0x00}},
    {"La", {0x70, 0xD4, 0xFF}},       {"Ce", {0xFF, 0xFF, 0xC7}},
    {"Pr", {0xD9, 0xFF, 0xC7}},       {"Nd", {0xC7, 0xFF, 0xC7}},
    {"Pm", {0xA3, 0xFF, 0xC7}},       {"Sm", {0x8F, 0xFF, 0xC7}},
    {"Eu", {0x61, 0xFF, 0xC7}},       {"Gd", {0x45, 0xFF, 0xC7}},
    {"Tb", {0x30, 0xFF, 0xC7}},       {"Dy", {0x1F, 0xFF, 0xC7}},
    {"Ho", {0x00, 0xFF, 0x9C}},       {"Er", {0x00, 0xE6, 0x75}},
    {"Tm", {0x00, 0xD4, 0x52}},       {"Yb", {0x00, 0xBF, 0x38}},
    {"Lu", {0x00, 0xAB, 0x24}},       {"Hf", {0x4D, 0xC2, 0xFF}},
    {"Ta", {0x4D, 0xA6, 0xFF}},       {"W", {0x21, 0x94, 0xD6}},
    {"Re", {0x26, 0x7D, 0xAB}},       {"Os", {0x26, 0x66, 0x96}},
    {"Ir", {0x17, 0x54, 0x87}},       {"Pt", {0xD0, 0xD0, 0xE0}},
    {"Au", {0xFF, 0xD1, 0x23}},       {"Hg", {0xB8, 0xB8, 0xD0}},
    {"Tl", {0xA6, 0x54, 0x4D}},       {"Pb", {0x57, 0x59, 0x61}},
    {"Bi", {0x9E, 0x4F, 0xB5}},       {"Po", {0xAB, 0x5C, 0x00}},
    {"At", {0x75, 0x4F, 0x45}},       {"Rn", {0x42, 0x82, 0x96}},
    {"Fr", {0x42, 0x00, 0x66}},       {"Ra", {0x00, 0x7D, 0x00}},
    {"Ac", {0x70, 0xAB, 0xFA}},       {"Th", {0x00, 0xBA, 0xFF}},
    {"Pa", {0x00, 0xA1, 0xFF}},       {"U", {0x00, 0x8F, 0xFF}},
    {"Np", {0x00, 0x80, 0xFF}},       {"Pu", {0x00, 0x6B, 0xFF}},
    {"Am", {0x54, 0x5C, 0xF2}},       {"Cm", {0x78, 0x5C, 0xE3}},
    {"Bk", {0x8A, 0x4F, 0xE3}},       {"Cf", {0xA1, 0x36, 0xD4}},
    {"Es", {0xB3, 0x1F, 0xD4}},       {"Fm", {0xB3, 0x1F, 0xBA}},
    {"Md", {0xB3, 0x0D, 0xA6}},       {"No", {0xBD, 0x0D, 0x87}},
    {"Lr", {0xC7, 0x00, 0x66}},       {"Rf", {0xCC, 0x00, 0x59}},
    {"Db", {0xD1, 0x00, 0x4F}},       {"Sg", {0xD9, 0x00, 0x45}},
    {"Bh", {0xE0, 0x00, 0x38}},       {"Hs", {0xE6, 0x00, 0x2E}},
    {"Mt", {0xEB, 0x00, 0x26}},       {"none", {0xFF, 0xFF, 0xFF}},
    {"selection", {0xFF, 0x00, 0x00}}};

Protein::Protein(brayns::Scene& scene, const ProteinDescriptor& descriptor)
    : _chainIds(descriptor.chainIds)
{
    std::ifstream file(descriptor.path.c_str());
    if (!file.is_open())
        throw std::runtime_error("Could not open " + descriptor.path);

    size_t lineIndex{0};

    while (file.good())
    {
        std::string line;
        std::getline(file, line);
        if (line.find("ATOM") == 0 || line.find("HETATM") == 0)
            _readAtom(line);
        else if (line.find("TITLE") == 0)
            _readTitle(line);
        else if (descriptor.loadBonds && line.find("CONECT") == 0)
            _readConnect(line);
        else if (line.find("SEQRES") == 0)
            _readSequence(line);
        //        else if (line.find("REMARK") == 0)
        //            _readRemark(line);
    }
    file.close();

    // Update sequences
    std::map<std::string, size_t> minSeqs;
    for (const auto& atom : _atomMap)
    {
        if (minSeqs.find(atom.second.chainId) == minSeqs.end())
            minSeqs[atom.second.chainId] = std::numeric_limits<size_t>::max();
        minSeqs[atom.second.chainId] =
            std::min(minSeqs[atom.second.chainId], atom.second.reqSeq);
    }
    for (const auto& minSeq : minSeqs)
        if (_sequenceMap.find(minSeq.first) != _sequenceMap.end())
        {
            auto& sequence = _sequenceMap[minSeq.first];
            for (size_t i = 0; i < minSeq.second; ++i)
                sequence.resNames[i] = ".";
        }

    auto model = scene.createModel();

    // Build 3d models according to atoms positions (re-centered to origin)
    brayns::Boxf bounds;

    for (const auto& atom : _atomMap)
        bounds.merge(atom.second.position);
    const auto& center = bounds.getCenter();
    for (auto& atom : _atomMap)
        atom.second.position -= center;

    _buildModel(*model, descriptor);

    // Metadata
    brayns::ModelMetadata metadata;
    metadata["Title"] = _title;
    metadata["Atoms"] = std::to_string(_atomMap.size());
    metadata["Bonds"] = std::to_string(_bondsMap.size());

    const auto& size = bounds.getSize();
    metadata["Size"] = std::to_string(size.x) + ", " + std::to_string(size.y) +
                       ", " + std::to_string(size.z) + " angstroms";

    for (const auto& sequence : getSequencesAsString())
        metadata["Amino Acid Sequence " + sequence.first] =
            "[" + std::to_string(sequence.second.size()) + "] " +
            sequence.second;

    _modelDescriptor =
        std::make_shared<brayns::ModelDescriptor>(std::move(model),
                                                  descriptor.name,
                                                  descriptor.path, metadata);
    // Transformation
    brayns::Transformation transformation;
    transformation.setRotationCenter(center);

    _modelDescriptor->setTransformation(transformation);

#if 0
    // Add cylinder
    const auto size = bounds.getSize();
    PLUGIN_ERROR << "size=" << size << std::endl;
    model->addCylinder(0, {{0, 0, 0}, {0, 0, size.z}, size.x});
#endif
}

void Protein::_buildModel(brayns::Model& model,
                          const ProteinDescriptor& descriptor)
{
    // Atoms
    for (const auto& atom : _atomMap)
    {
        auto material =
            model.createMaterial(atom.first, std::to_string(atom.first));
        const auto& rgb = atomColorMap[atom.second.element];
        material->setDiffuseColor(
            {rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f});
        model.addSphere(atom.first,
                        {atom.second.position,
                         descriptor.atomRadiusMultiplier * atom.second.radius});
    }

    // Bonds
    if (descriptor.loadBonds)
    {
        for (const auto& bond : _bondsMap)
        {
            const auto& atomSrc = _atomMap[bond.first];
            for (const auto bondedAtom : bond.second)
            {
                const auto& atomDest = _atomMap[bondedAtom];

                const auto center =
                    (atomDest.position + atomSrc.position) / 2.f;

                model.addCylinder(bond.first, {atomSrc.position, center,
                                               descriptor.atomRadiusMultiplier *
                                                   BOND_RADIUS});

                model.addCylinder(bondedAtom, {atomDest.position, center,
                                               descriptor.atomRadiusMultiplier *
                                                   BOND_RADIUS});
            }
        }
    }

    // Sticks
    if (descriptor.addSticks)
        for (const auto& atom1 : _atomMap)
            for (const auto& atom2 : _atomMap)
                if (atom1.first != atom2.first)
                {
                    const auto stick =
                        atom2.second.position - atom1.second.position;

                    if (length(stick) < DEFAULT_STICK_DISTANCE)
                    {
                        const auto center =
                            (atom2.second.position + atom1.second.position) /
                            2.f;
                        model.addCylinder(atom1.first,
                                          {atom1.second.position, center,
                                           descriptor.atomRadiusMultiplier *
                                               BOND_RADIUS});
                    }
                }
}

StringMap Protein::getSequencesAsString() const
{
    StringMap sequencesAsStrings;
    for (const auto& sequence : _sequenceMap)
    {
        std::string shortSequence;
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        sequencesAsStrings[sequence.first] = shortSequence;
        PLUGIN_DEBUG << sequence.first << " ("
                     << sequence.second.resNames.size()
                     << "): " << shortSequence << std::endl;
    }
    return sequencesAsStrings;
}

void Protein::setColorScheme(const ColorScheme& colorScheme,
                             const Palette& palette)
{
    switch (colorScheme)
    {
    case ColorScheme::none:
        for (auto& atom : _atomMap)
            _setMaterialDiffuseColor(atom.first, atomColorMap[0]);
        break;
    case ColorScheme::atoms:
        _setAtomColorScheme();
        break;
    case ColorScheme::chains:
        _setChainColorScheme(palette);
        break;
    case ColorScheme::residues:
        _setResiduesColorScheme(palette);
        break;
    case ColorScheme::amino_acid_sequence:
        _setAminoAcidSequenceColorScheme(palette);
        break;
    case ColorScheme::glycosylation_site:
        _setGlycosylationSiteColorScheme(palette);
        break;
    default:
        PLUGIN_THROW(std::runtime_error("Unknown colorscheme"))
    }
}

void Protein::_setAtomColorScheme()
{
    std::set<size_t> materialId;
    for (const auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(
            std::distance(atomColorMap.begin(),
                          atomColorMap.find(atom.second.element)));
        materialId.insert(index);

        _setMaterialDiffuseColor(atom.first, atomColorMap[atom.second.element]);
    }
    PLUGIN_INFO << "Applying atom color scheme (" << materialId.size() << ")"
                << std::endl;
}

void Protein::_setAminoAcidSequenceColorScheme(const Palette& palette)
{
    size_t atomCount = 0;
    for (const auto& sequence : _sequenceMap)
    {
        std::string shortSequence;
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        const auto sequencePosition = shortSequence.find(_aminoAcidSequence);
        if (sequencePosition != -1)
        {
            PLUGIN_INFO << _aminoAcidSequence << " was found at position "
                        << sequencePosition << std::endl;
            size_t minSeq = 1e6;
            size_t maxSeq = 0;
            for (auto& atom : _atomMap)
            {
                minSeq = std::min(minSeq, atom.second.reqSeq);
                maxSeq = std::max(maxSeq, atom.second.reqSeq);
                if (atom.second.reqSeq >= sequencePosition &&
                    atom.second.reqSeq <
                        sequencePosition + _aminoAcidSequence.length())
                {
                    _setMaterialDiffuseColor(atom.first, palette[1]);
                    ++atomCount;
                }
                else
                    _setMaterialDiffuseColor(atom.first, palette[0]);
            }
            PLUGIN_DEBUG << atomCount << "[" << minSeq << "," << maxSeq
                         << "] atoms where colored" << std::endl;
        }
        else
            PLUGIN_WARN << _aminoAcidSequence << " was not found in "
                        << shortSequence << std::endl;
    }
    PLUGIN_INFO << "Applying Amino Acid Sequence color scheme ("
                << (atomCount > 0 ? "2" : "1") << ")" << std::endl;
}

void Protein::_setGlycosylationSiteColorScheme(const Palette& palette)
{
    size_t totalNbSites = 0;

    // Initialize atom colors
    for (const auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(atom.second.chainId[0]) - 63;
        _setMaterialDiffuseColor(atom.first, palette[index]);
    }

    for (const auto& sequence : _sequenceMap)
    {
        size_t nbSites = 0;
        std::string shortSequence;
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        for (size_t i = 0; i < shortSequence.length(); ++i)
        {
            const char aminoAcid = shortSequence[i];
            if (aminoAcid == 'N')
            {
                if (i < shortSequence.length() - 2)
                {
                    const auto aminAcid1 = shortSequence[i + 1];
                    const auto aminAcid2 = shortSequence[i + 2];
                    if ((aminAcid2 == 'T' || aminAcid2 == 'S') &&
                        aminAcid1 != 'P')
                    {
                        ++nbSites;
                        for (const auto& atom : _atomMap)
                            if (atom.second.reqSeq == i)
                                _setMaterialDiffuseColor(atom.first,
                                                         palette[0]);
                    }
                }
            }
        }
        PLUGIN_INFO << "Found " << nbSites << " sites on sequence "
                    << sequence.first << std::endl;
        totalNbSites += nbSites;
    }
    PLUGIN_INFO << "Applying Glycosylation Site color scheme ("
                << (totalNbSites > 0 ? "2" : "1") << ")" << std::endl;
}

void Protein::_setChainColorScheme(const Palette& palette)
{
    std::set<size_t> materialId;
    for (auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(atom.second.chainId[0]) - 64;
        materialId.insert(index);
        _setMaterialDiffuseColor(atom.first, palette[index]);
    }
    PLUGIN_INFO << "Applying Chain color scheme (" << materialId.size() << ")"
                << std::endl;
}

void Protein::_setResiduesColorScheme(const Palette& palette)
{
    std::set<size_t> materialId;
    for (auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(
            std::distance(_residues.begin(),
                          _residues.find(atom.second.resName)));
        materialId.insert(index);
        _setMaterialDiffuseColor(atom.first, palette[index]);
    }
    PLUGIN_INFO << "Applying Residues color scheme (" << materialId.size()
                << ")" << std::endl;
}

void Protein::_setMaterialDiffuseColor(const size_t atomIndex,
                                       const RGBColor& color)
{
    auto& model = _modelDescriptor->getModel();
    auto material = model.getMaterial(atomIndex);
    if (material)
    {
        material->setDiffuseColor(
            {color.r / 255.f, color.g / 255.f, color.b / 255.f});
        material->commit();
    }
}

void Protein::_setMaterialDiffuseColor(const size_t atomIndex,
                                       const Color& color)
{
    auto& model = _modelDescriptor->getModel();
    auto material = model.getMaterial(atomIndex);
    if (material)
    {
        material->setDiffuseColor(color);
        material->commit();
    }
}

void Protein::_readAtom(const std::string& line)
{
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

    std::string s = line.substr(21, 1);
    std::string chainId = trim(s);
    if (!_loadChain(static_cast<size_t>(chainId[0] - 64)))
        return;

    const size_t serial = static_cast<size_t>(atoi(line.substr(6, 5).c_str()));

    auto& atom = _atomMap[serial];
    atom.chainId = chainId;

    s = line.substr(12, 4);
    atom.name = trim(s);

    s = line.substr(16, 1);
    atom.altLoc = trim(s);

    s = line.substr(17, 3);
    atom.resName = trim(s);

    _residues.insert(atom.resName);

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

    // Convert position from nanometers
    atom.position = 0.01f * atom.position;

    // Convert radius from angstrom
    atom.radius = 0.0001f * atomicRadii[atom.element];
}

void Protein::_readSequence(const std::string& line)
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

    Sequence& sequence = _sequenceMap[s];
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

void Protein::_readConnect(const std::string& line)
{
    // -------------------------------------------------------------------------
    // COLUMNS TYPE      FIELD    DEFINITION
    // -------------------------------------------------------------------------
    // 1 - 6   Record name "CONECT"
    // 7 - 11  Integer   serial Atom serial number
    // 12 - 16 Integer   serial Serial number of bonded atom
    // 17 - 21 Integer   serial Serial number of bonded atom
    // 22 - 26 Integer   serial Serial number of bonded atom
    // 27 - 31 Integer   serial Serial number of bonded atom
    // -------------------------------------------------------------------------

    const size_t serial = static_cast<size_t>(atoi(line.substr(6, 5).c_str()));

    if (_atomMap.find(serial) != _atomMap.end())
    {
        auto& bond = _bondsMap[serial];

        for (size_t i = 11; i < line.length(); i += 5)
        {
            std::string s = line.substr(i, 5);
            s = trim(s);
            if (!s.empty())
            {
                const size_t atomSerial = static_cast<size_t>(atoi(s.c_str()));
                if (_atomMap.find(atomSerial) != _atomMap.end())
                    bond.push_back(atomSerial);
            }
        }
    }
}

void Protein::_readRemark(const std::string& line)
{
    // -------------------------------------------------------------------------
    // COLUMNS TYPE      FIELD     DEFINITION
    // -------------------------------------------------------------------------
    // 1 - 6   Record name "REMARK"
    // 8 - 10  Integer   remarkNum Remark number. It is not an error for remark
    //                             n to exist in an entry when remark n-1 does
    //                             not.
    // 13 - 16 String    "ALN"
    // 17 - 18 String    "C"
    // 19 - 22 String    "TRG"
    // 23 - 81 String    Sequence
    // -------------------------------------------------------------------------

    std::string s = line.substr(9, 1);
    if (s != "3")
        return;

    if (line.length() < 23)
        return;

    s = line.substr(12, 3);
    if (s != "ALN")
        return;

    s = line.substr(16, 1);
    if (s != "C")
        return;

    s = line.substr(18, 3);
    if (s != "TRG")
        return;

    s = line.substr(22, line.length() - 23);
    Sequence& sequence = _sequenceMap[0];
    if (sequence.resNames.empty())
        sequence.resNames.push_back(s);
    else
        sequence.resNames[0] = sequence.resNames[0] + s;
}

void Protein::_readTitle(const std::string& line)
{
    std::string s = line.substr(11);
    _title = trim(s);
}

bool Protein::_loadChain(const size_t chainId)
{
    bool found = true;
    if (!_chainIds.empty())
    {
        found = false;
        for (const auto id : _chainIds)
        {
            if (id == chainId)
            {
                found = true;
                break;
            }
        }
    }
    return found;
}
