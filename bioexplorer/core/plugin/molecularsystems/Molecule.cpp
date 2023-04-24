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

#include "Molecule.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#ifdef USE_CGAL
#include <plugin/meshing/PointCloudMesher.h>
#include <plugin/meshing/SurfaceMesher.h>
#endif

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/io/MeshLoader.h>

#include <omp.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace brayns;
using namespace common;
#ifdef USE_CGAL
using namespace meshing;
#endif

const std::string METADATA_AA_RANGE = "Amino Acids range";
const std::string METADATA_AA_SEQUENCE = "Amino Acid Sequence";
const double DEFAULT_BOND_RADIUS = 0.025;
const double DEFAULT_STICK_DISTANCE = 0.185;

// Attempt to use signed distance field technique for molecules resulted in poor
// results. Disabled by default,but the code is still there and can be improved.
const bool DEFAULT_USE_SDF = false;
const Vector3f DEFAULT_SDF_DISPLACEMENT = {0.f, 0.f, 0.f};

// Atomic radii in picometers (10e-12 meters)
const double DEFAULT_ATOM_RADIUS = 25.0;
static AtomicRadii atomicRadii = {
    {{"H"}, {53.0}},   {{"HE"}, {31.0}},  {{"LI"}, {167.0}}, {{"BE"}, {112.0}},
    {{"B"}, {87.0}},   {{"C"}, {67.0}},   {{"N"}, {56.0}},   {{"O"}, {48.0}},
    {{"F"}, {42.0}},   {{"NE"}, {38.0}},  {{"NA"}, {190.0}}, {{"MG"}, {145.0}},
    {{"AL"}, {118.0}}, {{"SI"}, {111.0}}, {{"P"}, {98.0}},   {{"S"}, {88.0}},
    {{"CL"}, {79.0}},  {{"AR"}, {71.0}},  {{"K"}, {243.0}},  {{"CA"}, {194.0}},
    {{"SC"}, {184.0}}, {{"TI"}, {176.0}}, {{"V"}, {171.0}},  {{"CR"}, {166.0}},
    {{"MN"}, {161.0}}, {{"FE"}, {156.0}}, {{"CO"}, {152.0}}, {{"NI"}, {149.0}},
    {{"CU"}, {145.0}}, {{"ZN"}, {142.0}}, {{"GA"}, {136.0}}, {{"GE"}, {125.0}},
    {{"AS"}, {114.0}}, {{"SE"}, {103.0}}, {{"BR"}, {94.0}},  {{"KR"}, {88.0}},
    {{"RB"}, {265.0}}, {{"SR"}, {219.0}}, {{"Y"}, {212.0}},  {{"ZR"}, {206.0}},
    {{"NB"}, {198.0}}, {{"MO"}, {190.0}}, {{"TC"}, {183.0}}, {{"RU"}, {178.0}},
    {{"RH"}, {173.0}}, {{"PD"}, {169.0}}, {{"AG"}, {165.0}}, {{"CD"}, {161.0}},
    {{"IN"}, {156.0}}, {{"SN"}, {145.0}}, {{"SB"}, {133.0}}, {{"TE"}, {123.0}},
    {{"I"}, {115.0}},  {{"XE"}, {108.0}}, {{"CS"}, {298.0}}, {{"BA"}, {253.0}},
    {{"LA"}, {226.0}}, {{"CE"}, {210.0}}, {{"PR"}, {247.0}}, {{"ND"}, {206.0}},
    {{"PM"}, {205.0}}, {{"SM"}, {238.0}}, {{"EU"}, {231.0}}, {{"GD"}, {233.0}},
    {{"TB"}, {225.0}}, {{"DY"}, {228.0}}, {{"HO"}, {226.0}}, {{"ER"}, {226.0}},
    {{"TM"}, {222.0}}, {{"YB"}, {222.0}}, {{"LU"}, {217.0}}, {{"HF"}, {208.0}},
    {{"TA"}, {200.0}}, {{"W"}, {193.0}},  {{"RE"}, {188.0}}, {{"OS"}, {185.0}},
    {{"IR"}, {180.0}}, {{"PT"}, {177.0}}, {{"AU"}, {174.0}}, {{"HG"}, {171.0}},
    {{"TL"}, {156.0}}, {{"PB"}, {154.0}}, {{"BI"}, {143.0}}, {{"PO"}, {135.0}},
    {{"AT"}, {127.0}}, {{"RN"}, {120.0}}, {{"FR"}, {25.0}},  {{"RA"}, {25.0}},
    {{"AC"}, {25.0}},  {{"TH"}, {25.0}},  {{"PA"}, {25.0}},  {{"U"}, {25.0}},
    {{"NP"}, {25.0}},  {{"PU"}, {25.0}},  {{"AM"}, {25.0}},  {{"CM"}, {25.0}},
    {{"BK"}, {25.0}},  {{"CF"}, {25.0}},  {{"ES"}, {25.0}},  {{"FM"}, {25.0}},
    {{"MD"}, {25.0}},  {{"NO"}, {25.0}},  {{"LR"}, {25.0}},  {{"RF"}, {25.0}},
    {{"DB"}, {25.0}},  {{"SG"}, {25.0}},  {{"BH"}, {25.0}},  {{"HS"}, {25.0}},
    {{"MT"}, {25.0}},  {{"DS"}, {25.0}},  {{"RG"}, {25.0}},  {{"CN"}, {25.0}},
    {{"NH"}, {25.0}},  {{"FL"}, {25.0}},  {{"MC"}, {25.0}},  {{"LV"}, {25.0}},
    {{"TS"}, {25.0}},  {{"OG"}, {25.0}}};

Molecule::Molecule(Scene& scene, const size_ts& chainIds)
    : SDFGeometries()
    , _aminoAcidRange(std::numeric_limits<size_t>::max(),
                      std::numeric_limits<size_t>::min())
    , _scene(scene)
    , _chainIds(chainIds)
{
}

double Molecule::_getDisplacementValue(const DisplacementElement&)
{
    return 0.0;
}

void Molecule::_computeReqSetOffset()
{
    for (auto& sequence : _residueSequenceMap)
    {
        std::string physicalReqSeq;
        size_t firstReqSeq;
        size_t previousReqSeq;
        bool initialized{false};
        for (const auto& atom : _atomMap)
        {
            if (atom.second.chainId != sequence.first)
                continue;

            if (!initialized)
            {
                firstReqSeq = atom.second.reqSeq;
                previousReqSeq = firstReqSeq - 1;
                initialized = true;
            }

            if (previousReqSeq != atom.second.reqSeq)
            {
                if (atom.second.reqSeq != previousReqSeq + 1)
                    break;

                const auto it = aminoAcidMap.find(atom.second.resName);
                if (it != aminoAcidMap.end())
                    physicalReqSeq += it->second.shortName;

                if (physicalReqSeq.length() > 10)
                    break;
            }

            previousReqSeq = atom.second.reqSeq;
        }

        std::string theoreticalReqSeq;
        for (const auto& aa : sequence.second.resNames)
            theoreticalReqSeq += aminoAcidMap.find(aa)->second.shortName;

        sequence.second.offset =
            theoreticalReqSeq.find(physicalReqSeq) - firstReqSeq;
        PLUGIN_INFO(3, "Sequence [" << sequence.first
                                    << "], offset: " << sequence.second.offset
                                    << ", Theoretical: " << theoreticalReqSeq
                                    << ", Physical: " << physicalReqSeq);
    }
}

const StringMap Molecule::getSequencesAsString() const
{
    StringMap sequencesAsStrings;
    for (const auto& sequence : _residueSequenceMap)
    {
        std::string shortSequence = std::to_string(_aminoAcidRange.x) + "," +
                                    std::to_string(_aminoAcidRange.y) + ",";
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        sequencesAsStrings[sequence.first] = shortSequence;
        PLUGIN_DEBUG(sequence.first << " (" << sequence.second.resNames.size()
                                    << "): " << shortSequence);
    }
    return sequencesAsStrings;
}

void Molecule::_buildAtomicStruture(const ProteinRepresentation representation,
                                    const double atomRadiusMultiplier,
                                    const bool surface, const bool loadBonds,
                                    ThreadSafeContainer& container)
{
    const uint64_t userData = NO_USER_DATA;
    const bool useSdf = DEFAULT_USE_SDF;

    // Atoms
    std::map<uint64_t, Neighbours> neighbours;
    size_t currentReqSeq = 0;
    for (const auto& atom : _atomMap)
    {
        // Geometry
        const float radius =
            static_cast<float>(atom.second.radius * atomRadiusMultiplier);
        neighbours[currentReqSeq].insert(
            container.addSphere(atom.second.position, radius, atom.first,
                                useSdf, userData, neighbours[currentReqSeq],
                                DEFAULT_SDF_DISPLACEMENT));
        if (currentReqSeq != atom.second.reqSeq)
            currentReqSeq = atom.second.reqSeq;
    }

    // Bonds
    if (loadBonds)
    {
        PLUGIN_INFO(3, "Building " << _bondsMap.size() << " bonds...");
        for (const auto& bond : _bondsMap)
        {
            const auto& atomSrc = _atomMap.find(bond.first)->second;
            for (const auto bondedAtom : bond.second)
            {
                const auto& atomDst = _atomMap.find(bondedAtom)->second;
                const float radius = static_cast<float>(atomRadiusMultiplier *
                                                        DEFAULT_BOND_RADIUS);

                const auto center = (atomSrc.position + atomDst.position) / 2.0;

                const auto reqSeq = atomSrc.reqSeq;
                neighbours[reqSeq].insert(
                    container.addCone(atomSrc.position, radius, center, radius,
                                      bond.first, useSdf, userData,
                                      neighbours[reqSeq],
                                      DEFAULT_SDF_DISPLACEMENT));

                neighbours[reqSeq].insert(
                    container.addCone(atomDst.position, radius, center, radius,
                                      bondedAtom, useSdf, userData,
                                      neighbours[reqSeq],
                                      DEFAULT_SDF_DISPLACEMENT));
            }
        }
    }

    // Sticks
    if (representation == ProteinRepresentation::atoms_and_sticks)
    {
        PLUGIN_INFO(3, "Building sticks (" << _atomMap.size() << " atoms)...");
        auto it1 = _atomMap.begin();
        while (it1 != _atomMap.end())
        {
            const auto atom1 = (*it1);
            auto it2 = it1;
            ++it2;
            const auto reqSeq = atom1.second.reqSeq;
            while ((*it2).second.reqSeq == reqSeq)
            {
                const auto stick =
                    (*it2).second.position - atom1.second.position;
                if (length(stick) < DEFAULT_STICK_DISTANCE)
                {
                    const auto center =
                        ((*it2).second.position + atom1.second.position) / 2.0;
                    const float radius = static_cast<float>(
                        atomRadiusMultiplier * DEFAULT_BOND_RADIUS);
                    neighbours[reqSeq].insert(
                        container.addCone(atom1.second.position, radius, center,
                                          radius, atom1.first, useSdf, userData,
                                          neighbours[reqSeq],
                                          DEFAULT_SDF_DISPLACEMENT));
                    neighbours[reqSeq].insert(
                        container.addCone((*it2).second.position, radius,
                                          center, radius, (*it2).first, useSdf,
                                          userData, neighbours[reqSeq],
                                          DEFAULT_SDF_DISPLACEMENT));
                }
                ++it2;
                ++it1;
            }
            ++it1;
        }
    }
}

void Molecule::_rescaleMesh(Model& model, const Vector3f& scale)
{
    auto& triangleMeshes = model.getTriangleMeshes();
    const auto& bounds = model.getBounds();
    const Vector3f center = bounds.getCenter();
    for (auto& triangleMesh : triangleMeshes)
    {
        auto& vertices = triangleMesh.second.vertices;
        for (auto& vertex : vertices)
            vertex = center + (vertex - center) * scale;
    }
}

void Molecule::_buildModel(const std::string& assemblyName,
                           const std::string& name, const std::string& pdbId,
                           const std::string& header,
                           const ProteinRepresentation& representation,
                           const double atomRadiusMultiplier,
                           const bool loadBonds)
{
    PLUGIN_INFO(3, "Building protein " << name << " [PDB " << pdbId << "]...");

    // Metadata
    ModelMetadata metadata;
    metadata[METADATA_ASSEMBLY] = assemblyName;
    metadata[METADATA_PDB_ID] = pdbId;
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

    switch (representation)
    {
    case ProteinRepresentation::atoms:
    case ProteinRepresentation::atoms_and_sticks:
    {
        auto model = _scene.createModel();
        ThreadSafeContainer container(*model);

        _buildAtomicStruture(representation, atomRadiusMultiplier, false,
                             loadBonds, container);
        container.commitToModel();

        // Materials
        for (const auto atom : _atomMap)
        {
            const auto materialId = atom.first;
            auto material = model->getMaterial(materialId);
            RGBColorDetails rgb{255, 255, 255};
            const auto it = atomColorMap.find(atom.second.element);
            if (it != atomColorMap.end())
                rgb = (*it).second;

            brayns::PropertyMap props;
            props.setProperty(
                {MATERIAL_PROPERTY_CHAMELEON_MODE,
                 static_cast<int>(
                     MaterialChameleonMode::undefined_chameleon_mode)});
            props.setProperty(
                {MATERIAL_PROPERTY_NODE_ID, static_cast<int>(_uuid)});
            material->setDiffuseColor(
                {rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0});
            material->updateProperties(props);
        }

        _modelDescriptor =
            std::make_shared<ModelDescriptor>(std::move(model), name, header,
                                              metadata);

        break;
    }
    case ProteinRepresentation::mesh:
    {
        const std::string filename =
            GeneralSettings::getInstance()->getMeshFolder() + pdbId + ".obj";
        MeshLoader meshLoader(_scene);
        _modelDescriptor =
            meshLoader.importFromFile(filename, LoaderProgress(), {});
        _setMaterialExtraAttributes();
        _rescaleMesh(_modelDescriptor->getModel(), Vector3d(0.1, 0.1, 0.1));
        break;
    }
#ifdef USE_CGAL
    case ProteinRepresentation::surface:
    case ProteinRepresentation::union_of_balls:
    {
        // Surface
        Vector4ds pointCloud;
        const size_t materialId{0};
        for (const auto& atom : _atomMap)
        {
            if (atom.first % std::max(1, int(atomRadiusMultiplier)) != 0)
                continue;
            pointCloud.push_back({atom.second.position.x,
                                  atom.second.position.y,
                                  atom.second.position.z,
                                  atom.second.radius * atomRadiusMultiplier});
        }

        SurfaceMesher sm(_uuid);
        if (representation == ProteinRepresentation::union_of_balls)
            _modelDescriptor =
                sm.generateUnionOfBalls(_scene, pdbId, pointCloud);
        else
            _modelDescriptor = sm.generateSurface(_scene, pdbId, pointCloud);
        _setMaterialExtraAttributes();
        _modelDescriptor->setMetadata(metadata);

        Model& model = _modelDescriptor->getModel();
        ThreadSafeContainer container(model);
        _buildAtomicStruture(representation, atomRadiusMultiplier * 2.0, true,
                             loadBonds, container);
        container.commitToModel();
        // Materials
        for (const auto atom : _atomMap)
        {
            const auto materialId = atom.first;
            auto material = model.getMaterial(materialId);
            RGBColorDetails rgb{255, 255, 255};
            const auto it = atomColorMap.find(atom.second.element);
            if (it != atomColorMap.end())
                rgb = (*it).second;

            brayns::PropertyMap props;
            props.setProperty(
                {MATERIAL_PROPERTY_CHAMELEON_MODE,
                 static_cast<int>(MaterialChameleonMode::emitter)});
            props.setProperty(
                {MATERIAL_PROPERTY_NODE_ID, static_cast<int>(_uuid)});
            material->setDiffuseColor(
                {rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0});
            material->updateProperties(props);
        }
        break;
    }
    case ProteinRepresentation::contour:
    {
        auto model = _scene.createModel();
        PointCloud pointCloud;
        for (const auto& atom : _atomMap)
            pointCloud[0].push_back(
                {atom.second.position.x, atom.second.position.y,
                 atom.second.position.z,
                 atom.second.radius * atomRadiusMultiplier});

        PointCloudMesher pcm;
        ThreadSafeContainer container(*model);
        pcm.toConvexHull(container, pointCloud);
        container.commitToModel();
        _modelDescriptor =
            std::make_shared<ModelDescriptor>(std::move(model), name, header,
                                              metadata);
        _setMaterialExtraAttributes();
        break;
    }
#else
    case ProteinRepresentation::surface:
    case ProteinRepresentation::union_of_balls:
    case ProteinRepresentation::contour:
        PLUGIN_THROW("CGAL is required to create surfaces");
        break;
#endif
    case ProteinRepresentation::debug:
    {
        auto model = _scene.createModel();
        const size_t materialId = 0;
        brayns::Boxf box;
        for (const auto& atom : _atomMap)
            box.merge({atom.second.position.x, atom.second.position.y,
                       atom.second.position.z});

        const auto halfSize = box.getSize() * 0.5;
        const auto center = box.getCenter();

        const brayns::Vector3d a = {0.0, 0.0, center.z + halfSize.z};
        const brayns::Vector3d b = {0.0, 0.0, center.z - halfSize.z * 0.5};
        const brayns::Vector3d c = {0.0, 0.0, center.z - halfSize.z * 0.51};
        const brayns::Vector3d d = {0.0, 0.0, center.z - halfSize.z};

        model->addSphere(materialId,
                         {a, static_cast<float>(atomRadiusMultiplier * 0.2)});
        model->addCylinder(materialId,
                           {a, b,
                            static_cast<float>(atomRadiusMultiplier * 0.2)});
        model->addCone(materialId,
                       {b, c, static_cast<float>(atomRadiusMultiplier * 0.2),
                        static_cast<float>(atomRadiusMultiplier)});
        model->addCone(materialId,
                       {c, d, static_cast<float>(atomRadiusMultiplier), 0.0});
        _modelDescriptor =
            std::make_shared<ModelDescriptor>(std::move(model), name, header,
                                              metadata);
        break;
    }
    }

    PLUGIN_INFO(3, "Molecule model successfully built");

    PLUGIN_INFO(3, "---=== Molecule ===--- ");
    PLUGIN_INFO(3, "Assembly name         : " << assemblyName);
    PLUGIN_INFO(3, "Name                  : " << name);
    PLUGIN_INFO(3, "Atom Radius multiplier: " << atomRadiusMultiplier);
    PLUGIN_INFO(3, "Number of atoms       : " << _atomMap.size());
    PLUGIN_INFO(3, "Number of bonds       : " << _bondsMap.size());

    if (_modelDescriptor &&
        !GeneralSettings::getInstance()->getModelVisibilityOnCreation())
        _modelDescriptor->setVisible(false);
}

void Molecule::_readAtom(const std::string& line, const bool loadHydrogen)
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
    _aminoAcidRange.x = std::min(atom.reqSeq, size_t(_aminoAcidRange.x));
    _aminoAcidRange.y = std::max(atom.reqSeq, size_t(_aminoAcidRange.y));

    atom.iCode = line.substr(26, 1);

    atom.position.x = static_cast<double>(atof(line.substr(30, 8).c_str()));
    atom.position.y = static_cast<double>(atof(line.substr(38, 8).c_str()));
    atom.position.z = static_cast<double>(atof(line.substr(46, 8).c_str()));

    if (line.length() >= 60)
        atom.occupancy = static_cast<double>(atof(line.substr(54, 6).c_str()));

    if (line.length() >= 66)
        atom.tempFactor = static_cast<double>(atof(line.substr(60, 6).c_str()));

    if (line.length() >= 78)
    {
        s = line.substr(76, 2);
        atom.element = trim(s);
        if (s == "H" && !loadHydrogen)
            return;
    }

    if (line.length() >= 80)
    {
        s = line.substr(78, 2);
        atom.charge = trim(s);
    }

    // Convert position from amstrom (10e-10) to nanometers (10e-9)
    atom.position = 0.1 * atom.position;

    // Bounds
    _bounds.merge(atom.position);

    // Convert radius from picometers (10e-12) to nanometers (10e-9)
    atom.radius = 0.001 * DEFAULT_ATOM_RADIUS;
    auto it = atomicRadii.find(atom.element);
    if (it != atomicRadii.end())
        atom.radius = 0.001 * (*it).second;
    else
    {
        it = atomicRadii.find(atom.name);
        if (it != atomicRadii.end())
            atom.radius = 0.001 * (*it).second;
        else
            PLUGIN_DEBUG("[" << atom.element << "]/[" << atom.name
                             << "] not found");
    }

    _atomMap.insert(std::make_pair(serial, atom));
}

void Molecule::_readSequence(const std::string& line)
{
    // -------------------------------------------------------------------------
    // COLUMNS TYPE      FIELD    DEFINITION
    // -------------------------------------------------------------------------
    // 1 - 6   Record name "SEQRES"
    // 8 - 10  Integer   serNum   Serial number of the SEQRES record for the
    //                            current chain. Starts at 1 and increments
    //                            by one each line. Reset to 1 for each
    //                            chain.
    // 12      Character chainID  Chain identifier. This may be any single
    // legal
    //                            character, including a blank which is is
    //                            used if there is only one chain
    // 14 - 17 Integer   numRes   Number of residues in the chain. This
    // value is
    //                            repeated on every record.
    // 20 - 22 String    resName  Residue name
    // 24 - 26 ...
    // -------------------------------------------------------------------------

    std::string s = line.substr(11, 1);

    ResidueSequence& sequence = _residueSequenceMap[s];
    sequence.numRes = static_cast<size_t>(atoi(line.substr(13, 4).c_str()));

    for (size_t i = 19; i < line.length(); i += 4)
    {
        s = line.substr(i, 4);
        s = trim(s);
        if (!s.empty())
            sequence.resNames.push_back(s);
    }
}

void Molecule::_readConnect(const std::string& line)
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

void Molecule::_readRemark(const std::string& line)
{
    // -------------------------------------------------------------------------
    // COLUMNS TYPE      FIELD     DEFINITION
    // -------------------------------------------------------------------------
    // 1 - 6   Record name "REMARK"
    // 8 - 10  Integer   remarkNum Remark number. It is not an error for
    // remark
    //                             n to exist in an entry when remark n-1
    //                             does not.
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
    ResidueSequence& sequence = _residueSequenceMap[0];
    if (sequence.resNames.empty())
        sequence.resNames.push_back(s);
    else
        sequence.resNames[0] = sequence.resNames[0] + s;
}

std::string Molecule::_readTitle(const std::string& line)
{
    std::string s = line.substr(11);
    return trim(s);
}

std::string Molecule::_readHeader(const std::string& line)
{
    std::string s = line.substr(11);
    return trim(s);
}

bool Molecule::_loadChain(const size_t chainId)
{
    bool found = true;
    if (!_chainIds.empty())
    {
        found = false;
        for (const auto id : _chainIds)
            if (id == chainId)
            {
                found = true;
                break;
            }
    }
    return found;
}

void Molecule::_setAtomColorScheme()
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
    PLUGIN_INFO(3, "Applying atom color scheme (" << materialId.size() << ")");
}

void Molecule::_setAminoAcidSequenceColorScheme(const Palette& palette)
{
    if (palette.size() != 2)
        PLUGIN_THROW("Invalid palette size. 2 colors are expected");

    size_t atomCount = 0;
    for (const auto& sequence : _residueSequenceMap)
    {
        if (_selectedAminoAcidSequence.empty())
        {
            // Range based coloring
            for (auto& atom : _atomMap)
            {
                bool selected = false;
                for (const auto& range : _selectedAminoAcidRanges)
                {
                    selected = (atom.second.reqSeq >= range.x &&
                                atom.second.reqSeq <= range.y);
                    if (selected)
                        break;
                }
                _setMaterialDiffuseColor(atom.first,
                                         selected ? palette[1] : palette[0]);
            }
        }
        else
        {
            // String based coloring
            std::string shortSequence;
            for (const auto& resName : sequence.second.resNames)
                shortSequence += aminoAcidMap[resName].shortName;

            const auto sequencePosition =
                shortSequence.find(_selectedAminoAcidSequence);
            if (sequencePosition != -1)
            {
                PLUGIN_INFO(3, _selectedAminoAcidSequence
                                   << " was found at position "
                                   << sequencePosition);
                size_t minSeq = 1e6;
                size_t maxSeq = 0;
                for (auto& atom : _atomMap)
                {
                    minSeq = std::min(minSeq, atom.second.reqSeq);
                    maxSeq = std::max(maxSeq, atom.second.reqSeq);
                    if (atom.second.reqSeq >= sequencePosition &&
                        atom.second.reqSeq <
                            sequencePosition +
                                _selectedAminoAcidSequence.length())
                    {
                        _setMaterialDiffuseColor(atom.first, palette[1]);
                        ++atomCount;
                    }
                    else
                        _setMaterialDiffuseColor(atom.first, palette[0]);
                }
                PLUGIN_DEBUG(atomCount << "[" << minSeq << "," << maxSeq
                                       << "] atoms where colored");
            }
            else
                PLUGIN_WARN(_selectedAminoAcidSequence << " was not found in "
                                                       << shortSequence);
        }
    }
    PLUGIN_INFO(3, "Applying Amino Acid Sequence color scheme ("
                       << (atomCount > 0 ? "2" : "1") << ")");
}

void Molecule::_setChainColorScheme(const Palette& palette)
{
    std::set<size_t> materialId;
    for (auto& atom : _atomMap)
    {
        const size_t index = static_cast<size_t>(atom.second.chainId[0]) - 64;
        materialId.insert(index);
        _setMaterialDiffuseColor(atom.first, palette[index]);
    }
    PLUGIN_INFO(3, "Applying Chain color scheme (" << materialId.size() << ")");
}

void Molecule::_setResiduesColorScheme(const Palette& palette)
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
    PLUGIN_INFO(3,
                "Applying Residues color scheme (" << materialId.size() << ")");
}

void Molecule::_setMaterialDiffuseColor(const size_t atomIndex,
                                        const RGBColorDetails& color)
{
    auto& model = _modelDescriptor->getModel();
    auto material = model.getMaterial(atomIndex);
    if (material)
    {
        material->setDiffuseColor(
            {color.r / 255.0, color.g / 255.0, color.b / 255.0});
        material->commit();
    }
    else
        PLUGIN_THROW("Model has no material for atom " +
                     std::to_string(atomIndex));
}

void Molecule::_setMaterialDiffuseColor(const size_t atomIndex,
                                        const Color& color)
{
    auto& model = _modelDescriptor->getModel();
    try
    {
        auto material = model.getMaterial(atomIndex);
        if (material)
        {
            material->setDiffuseColor(color);
            material->markModified();
        }
        else
            PLUGIN_THROW("Model has no material for atom " +
                         std::to_string(atomIndex));
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
}
} // namespace molecularsystems
} // namespace bioexplorer