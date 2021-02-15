/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include "Protein.h"

#include <plugin/bioexplorer/Glycans.h>
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
        brayns::Boxf newBounds;
        const auto& center = _bounds.getCenter();
        for (auto& atom : _atomMap)
        {
            atom.second.position -= center;
            newBounds.merge(atom.second.position);
        }
        _bounds = newBounds;
    }

    _buildModel(_descriptor.assemblyName, _descriptor.name, title, header,
                _descriptor.representation, _descriptor.atomRadiusMultiplier,
                _descriptor.loadBonds);
}

Protein::~Protein()
{
    for (const auto& glycan : _glycans)
    {
        const auto modelId = glycan.second->getModelDescriptor()->getModelID();
        PLUGIN_INFO << "Removing glycan <" << modelId << "><" << glycan.first
                    << "> from assembly <" << _descriptor.name << ">"
                    << std::endl;
        _scene.removeModel(modelId);
    }
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
            const auto offsetSite = site + 1;
            bool validSite{false};
            Boxf siteBounds;
            std::string siteAtoms;
            for (const auto& atom : _atomMap)
                if (atom.second.chainId == chain.first &&
                    atom.second.reqSeq == site)
                {
                    siteBounds.merge(atom.second.position);
                    validSite = true;
                    siteAtoms += "{" + atom.second.name + "}";
                }

            if (validSite)
            {
                PLUGIN_DEBUG << "Chain: " << chain.first
                             << ", Site: " << offsetSite
                             << ", Atoms: " << siteAtoms << std::endl;
                const auto& proteinCenter = _bounds.getCenter();
                const auto& siteCenter = siteBounds.getCenter();

                const auto bindOrientation =
                    normalize(siteCenter - proteinCenter);

                positions.push_back(siteCenter);
                rotations.push_back(
                    glm::quatLookAt(bindOrientation, UP_VECTOR));
            }
            else
                PLUGIN_WARN << "Chain: " << chain.first
                            << ", Site: " << offsetSite
                            << ": no atoms available" << std::endl;
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

void Protein::setAminoAcid(const SetAminoAcid& aminoAcid)
{
    for (auto& sequence : _sequenceMap)
    {
        bool acceptChain = true;
        if (!aminoAcid.chainIds.empty())
        {
            const size_t chainId = static_cast<size_t>(sequence.first[0]) - 64;
            auto it = find(aminoAcid.chainIds.begin(), aminoAcid.chainIds.end(),
                           chainId);
            acceptChain = (it == aminoAcid.chainIds.end());
        }

        if (aminoAcid.index >= sequence.second.resNames.size())
            PLUGIN_THROW(std::runtime_error(
                "Invalid index for the amino acid sequence"));

        if (acceptChain)
            sequence.second.resNames[aminoAcid.index] =
                aminoAcid.aminoAcidShortName;
    }
}

void Protein::_processInstances(ModelDescriptorPtr md,
                                const Vector3fs& positions,
                                const Quaternions& orientations)
{
    size_t count = 0;
    const auto& proteinInstances = _modelDescriptor->getInstances();
    for (const auto& proteinInstance : proteinInstances)
    {
        const auto& proteinTransformation = proteinInstance.getTransformation();
        for (size_t i = 0; i < positions.size(); ++i)
        {
            const auto& position = positions[i];
            const auto& orientation = orientations[i];

            Transformation glycanTransformation;
            glycanTransformation.setTranslation(position);
            glycanTransformation.setRotation(orientation);

            const Transformation combinedTransformation =
                proteinTransformation * glycanTransformation;

            if (count == 0)
                md->setTransformation(combinedTransformation);

            const ModelInstance instance(true, false, combinedTransformation);
            md->addInstance(instance);
            ++count;
        }
    }
}

void Protein::addGlycans(const SugarsDescriptor& sd)
{
    Vector3fs glycanPositions;
    Quaternions glycanOrientations;
    getGlycosilationSites(glycanPositions, glycanOrientations, sd.siteIndices);

    if (glycanPositions.empty())
        PLUGIN_THROW(std::runtime_error("No glycosylation site was found on " +
                                        sd.proteinName));

    // Create glycans and attach them to the glycosylation sites of the target
    // protein
    GlycansPtr glycans(new Glycans(_scene, sd));
    auto modelDescriptor = glycans->getModelDescriptor();
    _processInstances(modelDescriptor, glycanPositions, glycanOrientations);

    _glycans[sd.name] = std::move(glycans);
    _scene.addModel(modelDescriptor);
}

void Protein::addSugars(const SugarsDescriptor& sd)
{
    Vector3fs positions;
    Quaternions orientations;
    getSugarBindingSites(positions, orientations, sd.siteIndices, sd.chainIds);

    if (positions.empty())
        PLUGIN_THROW(std::runtime_error("No sugar binding site was found on " +
                                        sd.name));

    PLUGIN_INFO << positions.size() << " sugar sites found on "
                << sd.proteinName << std::endl;

    GlycansPtr glucoses(new Glycans(_scene, sd));
    auto modelDescriptor = glucoses->getModelDescriptor();
    _processInstances(modelDescriptor, positions, orientations);

    _glycans[sd.name] = std::move(glucoses);
    _scene.addModel(modelDescriptor);
}

} // namespace bioexplorer