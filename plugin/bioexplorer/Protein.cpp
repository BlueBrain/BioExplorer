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

#include "Protein.h"

#include <plugin/common/Logs.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

#include <sstream>

namespace bioexplorer
{
Protein::Protein(Scene& scene, const ProteinDescriptor& descriptor)
    : Molecule(scene, descriptor.chainIds)
    , _descriptor(descriptor)
{
    size_t lineIndex{0};

    std::stringstream lines{_descriptor.contents};
    std::string line;
    std::string title{_descriptor.name};
    std::string header{_descriptor.name};

    while (getline(lines, line, '\n'))
    {
        if (line.find(KEY_ATOM) == 0)
            _readAtom(line, _descriptor.loadHydrogen);
        else if (descriptor.loadNonPolymerChemicals &&
                 line.find(KEY_HETATM) == 0)
            _readAtom(line, _descriptor.loadHydrogen);
        else if (line.find(KEY_HEADER) == 0)
            header = _readHeader(line);
        else if (line.find(KEY_TITLE) == 0)
            title = _readTitle(line);
        else if (descriptor.loadBonds && line.find(KEY_CONECT) == 0)
            _readConnect(line);
        else if (line.find(KEY_SEQRES) == 0)
            _readSequence(line);
        //        else if (line.find(KEY_REMARK) == 0)
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

    // Build 3d models according to atoms positions (re-centered to origin)
    if (descriptor.recenter)
    {
        const auto& center = _bounds.getCenter();
        for (auto& atom : _atomMap)
            atom.second.position -= center;
    }

    _buildModel(_descriptor.assemblyName, _descriptor.name, title, header,
                _descriptor.representation, _descriptor.atomRadiusMultiplier,
                _descriptor.loadBonds);
}

void Protein::setColorScheme(const ColorScheme& colorScheme,
                             const Palette& palette, const size_ts& chainIds)
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
    case ColorScheme::region:
        _setRegionColorScheme(palette, chainIds);
        break;
    default:
        PLUGIN_THROW(std::runtime_error("Unknown colorscheme"))
    }
}

void Protein::_setRegionColorScheme(const Palette& palette,
                                    const size_ts& chainIds)
{
    size_t atomCount = 0;
    for (auto& atom : _atomMap)
    {
        bool applyColor{true};
        if (!chainIds.empty())
        {
            const size_t chainId =
                static_cast<size_t>(atom.second.chainId[0] - 64);
            applyColor = (std::find(chainIds.begin(), chainIds.end(),
                                    chainId) != chainIds.end());
        }
        if (applyColor)
            _setMaterialDiffuseColor(atom.first, palette[atom.second.reqSeq]);
    }

    PLUGIN_INFO << "Applying Amino Acid Sequence color scheme ("
                << (atomCount > 0 ? "2" : "1") << ")" << std::endl;
} // namespace bioexplorer

void Protein::_setGlycosylationSiteColorScheme(const Palette& palette)
{
    if (palette.size() != 2)
        PLUGIN_THROW(
            std::runtime_error("Invalid palette size. 2 colors are expected"));

    // Initialize atom colors
    for (const auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(atom.second.chainId[0]) - 63;
        _setMaterialDiffuseColor(atom.first, palette[0]);
    }

    const auto sites = getGlycosylationSites({});

    for (const auto chain : sites)
        for (const auto site : chain.second)
            for (const auto& atom : _atomMap)
                if (atom.second.chainId == chain.first &&
                    atom.second.reqSeq == site)
                    _setMaterialDiffuseColor(atom.first, palette[1]);

    PLUGIN_INFO << "Applying Glycosylation Site color scheme ("
                << (sites.size() > 0 ? "2" : "1") << ")" << std::endl;
}

std::map<std::string, size_ts> Protein::getGlycosylationSites(
    const std::vector<size_t>& siteIndices) const
{
    std::map<std::string, size_ts> sites;
    for (const auto& sequence : _sequenceMap)
    {
        std::string shortSequence;
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        for (size_t i = 0; i < shortSequence.length(); ++i)
        {
            bool acceptSite{true};
            if (!siteIndices.empty())
            {
                const auto it =
                    find(siteIndices.begin(), siteIndices.end(), i + 1);
                acceptSite = (it != siteIndices.end());
            }

            const char aminoAcid = shortSequence[i];
            if (aminoAcid == 'N' && acceptSite)
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
    {
        std::string indices = "[";
        for (const auto& index : site.second)
        {
            if (indices.length() > 1)
                indices += ",";
            indices += std::to_string(index + 1); // Indices start at 1, not 0
        }
        indices += "]";
        PLUGIN_INFO << "Found " << site.second.size() << " glycosylation sites "
                    << indices << " on sequence " << site.first << std::endl;
    }
    return sites;
}

void Protein::_getSitesTransformations(
    std::vector<Vector3f>& positions, std::vector<Quaterniond>& rotations,
    const std::map<std::string, size_ts>& sites) const
{
    for (const auto chain : sites)
    {
        for (const auto site : chain.second)
        {
            bool validSite{false};
            Boxf bounds;
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
                    glm::quatLookAt(normalize(center - _bounds.getCenter()),
                                    {0.f, 0.f, -1.f}));
            }
        }
    }
}

void Protein::getGlycosilationSites(std::vector<Vector3f>& positions,
                                    std::vector<Quaterniond>& rotations,
                                    const size_ts& siteIndices) const
{
    positions.clear();
    rotations.clear();

    const auto sites = getGlycosylationSites(siteIndices);

    _getSitesTransformations(positions, rotations, sites);
}

void Protein::getSugarBindingSites(std::vector<Vector3f>& positions,
                                   std::vector<Quaterniond>& rotations,
                                   const size_ts& siteIndices,
                                   const size_ts& chainIds) const
{
    positions.clear();
    rotations.clear();

    std::set<std::string> chainIdsAsString;
    for (const auto& atom : _atomMap)
    {
        bool acceptChain{true};
        const size_t chainId = static_cast<size_t>(atom.second.chainId[0] - 64);
        if (!chainIds.empty())
            acceptChain = (std::find(chainIds.begin(), chainIds.end(),
                                     chainId) != chainIds.end());

        if (acceptChain)
            chainIdsAsString.insert(atom.second.chainId);
    }

    std::map<std::string, size_ts> sites;
    for (const auto& chainIdAsString : chainIdsAsString)
        sites[chainIdAsString] = siteIndices;

    _getSitesTransformations(positions, rotations, sites);
}

} // namespace bioexplorer
