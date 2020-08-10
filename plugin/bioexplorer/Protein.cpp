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

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/meshing/PointCloudMesher.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
const std::string METADATA_AA_RANGE = "Amino acids range";
const std::string METADATA_AA_SEQUENCE = "Amino Acid Sequence";

Protein::Protein(Scene& scene, const ProteinDescriptor& descriptor)
    : Molecule(descriptor.chainIds)
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
            for (size_t i = 1;
                 i < minSeq.second && i < sequence.resNames.size(); ++i)
                sequence.resNames.insert(sequence.resNames.begin(), ".");
        }
#endif

    auto model = scene.createModel();

    // Build 3d models according to atoms positions (re-centered to origin)
    if (descriptor.recenter)
    {
        const auto& center = _bounds.getCenter();
        for (auto& atom : _atomMap)
            atom.second.position -= center;
    }

    _buildModel(*model, descriptor);

    // Metadata
    ModelMetadata metadata;
    metadata[METADATA_ASSEMBLY] = descriptor.assemblyName;
    metadata[METADATA_TITLE] = title;
    metadata[METADATA_HEADER] = header;
    metadata[METADATA_ATOMS] = std::to_string(_atomMap.size());
    metadata[METADATA_BONDS] = std::to_string(_bondsMap.size());
    metadata[METADATA_AA_RANGE] = std::to_string(_aminoAcidRange.x) + ":" +
                                  std::to_string(_aminoAcidRange.y);

    const auto& size = _bounds.getSize();
    metadata[METADATA_SIZE] = std::to_string(size.x) + ", " +
                              std::to_string(size.y) + ", " +
                              std::to_string(size.z) + " angstroms";

    for (const auto& sequence : getSequencesAsString())
        metadata[METADATA_AA_SEQUENCE + sequence.first] =
            "[" + std::to_string(sequence.second.size()) + "] " +
            sequence.second;

    _modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), descriptor.name,
                                          header, metadata);

    PLUGIN_INFO << "---===  Protein  ===--- " << std::endl;
    PLUGIN_INFO << "Assembly name         : " << _descriptor.assemblyName
                << std::endl;
    PLUGIN_INFO << "Name                  : " << _descriptor.name << std::endl;
    PLUGIN_INFO << "Adom Radius multiplier: "
                << _descriptor.atomRadiusMultiplier << std::endl;
    PLUGIN_INFO << "Number of atoms       : " << _atomMap.size() << std::endl;
    PLUGIN_INFO << "Number of bonds       : " << _bondsMap.size() << std::endl;
    PLUGIN_INFO << "Position              : " << _descriptor.position[0] << ","
                << _descriptor.position[1] << "," << _descriptor.position[2]
                << std::endl;
    PLUGIN_INFO << "Orientation           : " << _descriptor.orientation[0]
                << "," << _descriptor.orientation[1] << ","
                << _descriptor.orientation[2] << ","
                << _descriptor.orientation[3] << std::endl;
}

void Protein::_buildModel(Model& model, const ProteinDescriptor& descriptor)
{
    // Atoms
    PLUGIN_INFO << "Building protein " << descriptor.name << "..." << std::endl;
    switch (descriptor.representation)
    {
    case ProteinRepresentation::atoms:
    case ProteinRepresentation::atoms_and_sticks:
    {
        for (const auto& atom : _atomMap)
        {
            auto material =
                model.createMaterial(atom.first, std::to_string(atom.first));

            RGBColor rgb{255, 255, 255};
            const auto it = atomColorMap.find(atom.second.element);
            if (it != atomColorMap.end())
                rgb = (*it).second;

            brayns::PropertyMap props;
            props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                               static_cast<int>(MaterialShadingMode::basic)});
            props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
            material->setDiffuseColor(
                {rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f});
            material->updateProperties(props);
            model.addSphere(atom.first, {atom.second.position,
                                         descriptor.atomRadiusMultiplier *
                                             atom.second.radius});
        }
        break;
    }
    case ProteinRepresentation::contour:
    {
        PointCloudMesher pcm;
        PointCloud pointCloud;
        for (const auto& atom : _atomMap)
            pointCloud[0].push_back(
                {atom.second.position.x, atom.second.position.y,
                 atom.second.position.z, atom.second.radius});
        pcm.toConvexHull(model, pointCloud);
        break;
    }
    case ProteinRepresentation::surface:
    {
        PointCloudMesher pcm;
        PointCloud pointCloud;
        const size_t materialId{0};

        auto material = model.createMaterial(materialId, "Metaball");
        brayns::PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::diffuse)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        material->setDiffuseColor({1.f, 1.f, 1.f});
        material->updateProperties(props);

        for (const auto& atom : _atomMap)
            pointCloud[materialId].push_back(
                {atom.second.position.x, atom.second.position.y,
                 atom.second.position.z, atom.second.radius});
        const size_t gridSize = 50;
        const float threshold = 10.0f;

        PLUGIN_INFO << "Metaballing " << gridSize
                    << ", threshold = " << threshold << std::endl;
        pcm.toMetaballs(model, pointCloud, gridSize, threshold);
        break;
    }
    }

    // Bonds
    if (descriptor.loadBonds)
    {
        PLUGIN_INFO << "Building " << _bondsMap.size() << " bonds..."
                    << std::endl;
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
    if (descriptor.representation == ProteinRepresentation::atoms_and_sticks)
    {
        PLUGIN_INFO << "Building sticks (" << _atomMap.size() << " atoms)..."
                    << std::endl;
        for (const auto& atom1 : _atomMap)
            for (const auto& atom2 : _atomMap)
                if (atom1.first != atom2.first &&
                    atom1.second.reqSeq == atom2.second.reqSeq &&
                    atom1.second.chainId == atom2.second.chainId)
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
    PLUGIN_INFO << "Protein model successfully built" << std::endl;
}

StringMap Protein::getSequencesAsString() const
{
    StringMap sequencesAsStrings;
    for (const auto& sequence : _sequenceMap)
    {
        std::string shortSequence = std::to_string(_aminoAcidRange.x) + "," +
                                    std::to_string(_aminoAcidRange.y) + ",";
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
    // Initialize atom colors
    for (const auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(atom.second.chainId[0]) - 63;
        _setMaterialDiffuseColor(atom.first, palette[index]);
    }

    const auto sites = getGlycosylationSites({});

    for (const auto chain : sites)
        for (const auto site : chain.second)
            for (const auto& atom : _atomMap)
                if (atom.second.chainId == chain.first &&
                    atom.second.reqSeq == site)
                    _setMaterialDiffuseColor(atom.first, palette[0]);

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

void Protein::getGlucoseBindingSites(std::vector<Vector3f>& positions,
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
