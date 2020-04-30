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

Protein::Protein(brayns::Scene& scene, const ProteinDescriptor& descriptor)
    : Node()
    , _chainIds(descriptor.chainIds)
    , _descritpor(descriptor)
{
    size_t lineIndex{0};

    std::stringstream lines{descriptor.contents};
    std::string line;

    while (getline(lines, line, '\n'))
    {
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

    if (_sequenceMap.empty())
    {
        // Build AA sequences from ATOMS if not SEQRES record exists
        size_t previousReqSeq = std::numeric_limits<size_t>::max();
        for (const auto& atom : _atomMap)
        {
            auto& sequence = _sequenceMap[atom.second.chainId];
            if (previousReqSeq != atom.second.reqSeq)
                sequence.resNames.push_back(atom.second.resName);
            previousReqSeq = atom.second.reqSeq;
        }
        for (auto& sequence : _sequenceMap)
            sequence.second.numRes = sequence.second.resNames.size();
    }

#if 0
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
            for (size_t i = 0;
                 i < minSeq.second && i < sequence.resNames.size(); ++i)
                sequence.resNames[i] = ".";
        }
#endif

    auto model = scene.createModel();

    // Build 3d models according to atoms positions (re-centered to origin)
    brayns::Boxf bounds;

    // Recenter
    if (descriptor.recenter)
    {
        for (const auto& atom : _atomMap)
            bounds.merge(atom.second.position);
        const auto& center = bounds.getCenter();
        for (auto& atom : _atomMap)
            atom.second.position -= center;
    }

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

    _modelDescriptor = std::make_shared<brayns::ModelDescriptor>(
        std::move(model), descriptor.name, descriptor.contents, metadata);
}

void Protein::_buildModel(brayns::Model& model,
                          const ProteinDescriptor& descriptor)
{
    // Atoms
    for (const auto& atom : _atomMap)
    {
        auto material =
            model.createMaterial(atom.first, std::to_string(atom.first));

        RGBColor rgb{255, 255, 255};
        const auto it = atomColorMap.find(atom.second.element);
        if (it != atomColorMap.end())
            rgb = (*it).second;

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
            const auto& atomSrc = _atomMap.find(bond.first)->second;
            for (const auto bondedAtom : bond.second)
            {
                const auto& atomDest = _atomMap.find(bondedAtom)->second;

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
        if (_aminoAcidSequence.empty())
        {
            // Range based coloring
            for (auto& atom : _atomMap)
                _setMaterialDiffuseColor(
                    atom.first, (atom.second.reqSeq >= _aminoAcidRange.x &&
                                 atom.second.reqSeq <= _aminoAcidRange.y)
                                    ? palette[1]
                                    : palette[0]);
        }
        else
        {
            // String based coloring
            std::string shortSequence;
            for (const auto& resName : sequence.second.resNames)
                shortSequence += aminoAcidMap[resName].shortName;

            const auto sequencePosition =
                shortSequence.find(_aminoAcidSequence);
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
    }
    PLUGIN_INFO << "Applying Amino Acid Sequence color scheme ("
                << (atomCount > 0 ? "2" : "1") << ")" << std::endl;
}

void Protein::_setGlycosylationSiteColorScheme(const Palette& palette)
{
    // Initialize atom colors
    for (const auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(atom.second.chainId[0]) - 63;
        _setMaterialDiffuseColor(atom.first, palette[index]);
    }

    const auto sites = _getGlycosylationSites();

    for (const auto chain : sites)
        for (const auto site : chain.second)
            for (const auto& atom : _atomMap)
                if (atom.second.chainId == chain.first &&
                    atom.second.reqSeq == site)
                    _setMaterialDiffuseColor(atom.first, palette[0]);

    PLUGIN_INFO << "Applying Glycosylation Site color scheme ("
                << (sites.size() > 0 ? "2" : "1") << ")" << std::endl;
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

    Atom atom;
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
    atom.radius = DEFAULT_ATOM_RADIUS;
    auto it = atomicRadii.find(atom.element);
    if (it != atomicRadii.end())
        atom.radius = 0.0001f * (*it).second;
    else
    {
        it = atomicRadii.find(atom.name);
        if (it != atomicRadii.end())
            atom.radius = 0.0001f * (*it).second;
    }

    _atomMap.insert(std::make_pair(serial, atom));
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
    //    sequence.serNum = static_cast<size_t>(atoi(line.substr(7,
    //    3).c_str()));
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
        auto& bond = _bondsMap.find(serial)->second;

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

std::map<std::string, size_ts> Protein::_getGlycosylationSites() const
{
    std::map<std::string, size_ts> sites;
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
                        sites[sequence.first].push_back(i);
                }
            }
        }
    }
    for (const auto& site : sites)
        PLUGIN_INFO << "Found " << site.second.size()
                    << " glycosylation on sequence " << site.first << std::endl;
    return sites;
}

void Protein::getGlycosilationSites(
    std::vector<brayns::Vector3f>& positions,
    std::vector<brayns::Quaterniond>& rotations) const
{
    positions.clear();
    rotations.clear();

    const auto sites = _getGlycosylationSites();
    for (const auto chain : sites)
    {
        for (const auto site : chain.second)
        {
            bool validSite{false};
            brayns::Boxf bounds;
            for (const auto& atom : _atomMap)
                if (atom.second.chainId == chain.first &&
                    atom.second.reqSeq == site)
                {
                    bounds.merge(atom.second.position);
                    validSite = true;
                }

            if (validSite)
            {
                const auto& center = bounds.getCenter();
                positions.push_back(center);
                rotations.push_back(
                    glm::quatLookAt(normalize(center), {0.f, 1.f, 0.f}));
            }
        }
    }
}
