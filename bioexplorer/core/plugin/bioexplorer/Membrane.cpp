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

#include "Membrane.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>

namespace bioexplorer
{
Membrane::Membrane(Scene &scene, const MembraneDescriptor &descriptor,
                   const Vector3f &position, const Quaterniond &orientation,
                   const Vector4fs &clippingPlanes,
                   const OccupiedDirections &occupiedDirections)
    : _scene(scene)
    , _position(position)
    , _orientation(orientation)
    , _descriptor(descriptor)
    , _clippingPlanes(clippingPlanes)
    , _occupiedDirections(occupiedDirections)
{
    if (_descriptor.representation == ProteinRepresentation::contour)
    {
        switch (_descriptor.shape)
        {
        case AssemblyShape::spherical:
            const std::string name{"Membrane"};
            const size_t materialId = 0;
            auto model = _scene.createModel();

            auto material = model->createMaterial(materialId, name);
            brayns::PropertyMap props;
            props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                               static_cast<int>(MaterialShadingMode::basic)});
            props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
            props.setProperty(
                {MATERIAL_PROPERTY_CHAMELEON_MODE,
                 static_cast<int>(
                     MaterialChameleonMode::undefined_chameleon_mode)});
            material->updateProperties(props);

            model->addSphere(materialId,
                             {_position, _descriptor.assemblyParams[0]});
            auto modelDescriptor =
                std::make_shared<ModelDescriptor>(std::move(model), name);
            _scene.addModel(modelDescriptor);
            return;
        }
    }

    std::vector<std::string> proteinContents;
    proteinContents.push_back(descriptor.content1);
    if (!descriptor.content2.empty())
        proteinContents.push_back(descriptor.content2);
    if (!descriptor.content3.empty())
        proteinContents.push_back(descriptor.content3);
    if (!descriptor.content4.empty())
        proteinContents.push_back(descriptor.content4);

    // Load proteins
    size_t i = 0;
    for (const auto &content : proteinContents)
    {
        ProteinDescriptor pd;
        pd.assemblyName = descriptor.assemblyName;
        pd.name = _getElementNameFromId(i);
        pd.contents = content;
        pd.chainIds = descriptor.chainIds;
        pd.recenter = descriptor.recenter;
        pd.loadBonds = descriptor.loadBonds;
        pd.randomSeed = descriptor.randomSeed;
        pd.occurrences = 1;
        pd.orientation = descriptor.orientation;
        pd.assemblyParams = descriptor.assemblyParams;
        pd.atomRadiusMultiplier = descriptor.atomRadiusMultiplier;
        pd.representation = descriptor.representation;
        pd.locationCutoffAngle = descriptor.locationCutoffAngle;
        pd.loadNonPolymerChemicals = descriptor.loadNonPolymerChemicals;
        pd.positionRandomizationType = descriptor.positionRandomizationType;
        pd.position = {0.f, 0.f, 0.f};

        ProteinPtr protein(new Protein(_scene, pd));
        auto modelDescriptor = protein->getModelDescriptor();
        _proteins[pd.name] = std::move(protein);
        ++i;
    }

    // Assemble proteins
    _processInstances();

    // Add proteins to the scene
    for (size_t i = 0; i < proteinContents.size(); ++i)
        _scene.addModel(
            _proteins[_getElementNameFromId(i)]->getModelDescriptor());
}

Membrane::~Membrane()
{
    for (const auto &protein : _proteins)
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
}

void Membrane::_processInstances()
{
    // Randomization
    srand(_descriptor.randomSeed);
    size_t rnd{1};
    if (_descriptor.randomSeed != 0 && _descriptor.positionRandomizationType ==
                                           PositionRandomizationType::circular)
        rnd = rand() % _descriptor.occurrences;

    const Quaterniond orientation = {_descriptor.orientation[0],
                                     _descriptor.orientation[1],
                                     _descriptor.orientation[2],
                                     _descriptor.orientation[3]};
    std::map<size_t, size_t> instanceCounts;
    for (size_t i = 0; i < _proteins.size(); ++i)
        instanceCounts[i] = 0;

    // Shape parameters
    const auto &params = _descriptor.assemblyParams;
    if (params.size() < 6)
        PLUGIN_THROW(std::runtime_error("Invalid number of shape parameters"));

    const float size = params[0];
    const float rndPosStength = params[2];
    const float rndDirStrength = params[4];
    const float extraParameter = params[5];

    // Shape instances
    const float offset = 2.f / _descriptor.occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    for (uint64_t i = 0; i < _descriptor.occurrences; ++i)
    {
        const size_t id = rand() % _proteins.size();
        const auto name = _getElementNameFromId(id);
        if (_proteins.find(name) == _proteins.end())
        {
            PLUGIN_ERROR << "Protein " << name << " is not registered"
                         << std::endl;
            continue;
        }
        auto protein = _proteins[name];
        auto md = protein->getModelDescriptor();

        const auto &model = md->getModel();
        const auto &bounds = model.getBounds();
        const Vector3f &center = bounds.getCenter();

        const size_t rndPosSeed = (params[1] == 0 ? 0 : params[1] + i);
        const size_t rndDirSeed = (params[3] == 0 ? 0 : params[3] + i);

        Vector3f pos;
        Vector3f dir{0.f, 1.f, 0.f};
        switch (_descriptor.shape)
        {
        case AssemblyShape::spherical:
        {
            getSphericalPosition(rnd, size, rndPosStength,
                                 _descriptor.positionRandomizationType,
                                 _descriptor.randomSeed, i,
                                 _descriptor.occurrences, Vector3f(), pos, dir);
            break;
        }
        case AssemblyShape::sinusoidal:
        {
            getSinosoidalPosition(size, extraParameter,
                                  _descriptor.positionRandomizationType,
                                  rndPosSeed, rndPosStength, rndDirSeed,
                                  rndDirStrength, Vector3f(), pos, dir);
            break;
        }
        case AssemblyShape::cubic:
        {
            getCubicPosition(size, Vector3f(), rndPosSeed, rndPosStength,
                             rndDirSeed, rndDirStrength, pos, dir);
            break;
        }
        case AssemblyShape::fan:
        {
            getFanPosition(rnd, size, _descriptor.positionRandomizationType,
                           _descriptor.randomSeed, i, _descriptor.occurrences,
                           Vector3f(), pos, dir);
            break;
        }
        case AssemblyShape::bezier:
        {
            const Vector3fs points = {
                {1, 391, 0},   {25, 411, 0},  {48, 446, 0},  {58, 468, 0},
                {70, 495, 0},  {83, 523, 0},  {110, 535, 0}, {157, 517, 0},
                {181, 506, 0}, {214, 501, 0}, {216, 473, 0}, {204, 456, 0},
                {223, 411, 0}, {241, 382, 0}, {261, 372, 0}, {297, 402, 0},
                {308, 433, 0}, {327, 454, 0}, {355, 454, 0}, {389, 446, 0},
                {406, 433, 0}, {431, 426, 0}, {458, 443, 0}, {478, 466, 0},
                {518, 463, 0}, {559, 464, 0}, {584, 478, 0}, {582, 503, 0},
                {550, 533, 0}, {540, 550, 0}, {540, 574, 0}, {560, 572, 0},
                {599, 575, 0}, {629, 550, 0}, {666, 548, 0}, {696, 548, 0},
                {701, 582, 0}, {701, 614, 0}, {683, 639, 0}, {653, 647, 0},
                {632, 651, 0}, {597, 666, 0}, {570, 701, 0}, {564, 731, 0},
                {559, 770, 0}, {565, 799, 0}, {577, 819, 0}, {611, 820, 0},
                {661, 809, 0}, {683, 787, 0}, {700, 768, 0}, {735, 758, 0},
                {763, 768, 0}, {788, 792, 0}, {780, 820, 0}, {770, 859, 0},
                {740, 882, 0}, {705, 911, 0}, {688, 931, 0}, {646, 973, 0},
                {611, 992, 0}, {585, 1022, 0}};
            getBezierPosition(points, size,
                              float(i) / float(_descriptor.occurrences), pos,
                              dir);
            break;
        }
        case AssemblyShape::spherical_to_planar:
        {
            getSphericalToPlanarPosition(rnd, size, rndPosStength,
                                         _descriptor.positionRandomizationType,
                                         _descriptor.randomSeed, i,
                                         _descriptor.occurrences, Vector3f(),
                                         extraParameter, pos, dir);
            break;
        }
        default:
            getPlanarPosition(size, _descriptor.positionRandomizationType,
                              _descriptor.randomSeed, Vector3f(), pos, dir);
            break;
        }

        // Remove membrane where proteins are. This is currently done
        // according to the vector orientation
        bool occupied{false};
        if (_descriptor.locationCutoffAngle != 0.f)
            for (const auto &occupiedDirection : _occupiedDirections)
                if (dot(dir, occupiedDirection.first) >
                    occupiedDirection.second)
                {
                    occupied = true;
                    break;
                }
        if (occupied)
            continue;

        // Final transformation
        Transformation tf;
        const Vector3f translation =
            _position + Vector3f(_orientation * Vector3d(pos - center));
        tf.setTranslation(translation);

        Quaterniond instanceOrientation = glm::quatLookAt(dir, UP_VECTOR);

#if 1
        tf.setRotation(_orientation * instanceOrientation * orientation);
#else
        if (_descriptor.randomSeed == 0)
            tf.setRotation(_orientation * instanceOrientation * orientation);
        else
        {
            // Add a bit of randomness in the orientation of the proteins
            Vector3f eulerAngles(0.3 * rnd(), 0.3 * rnd(), 0.3 * rnd());
            Quaterniond randomOrientation = glm::quat(eulerAngles);

            tf.setRotation(_orientation * instanceOrientation * orientation *
                           randomOrientation);
        }
#endif

        if (instanceCounts[id] == 0)
            md->setTransformation(tf);
        const ModelInstance instance(true, false, tf);
        md->addInstance(instance);

        instanceCounts[id] = instanceCounts[id] + 1;
    }

    for (size_t i = 0; i < _proteins.size(); ++i)
        PLUGIN_INFO << "Instances for " << i << " : " << instanceCounts[i]
                    << std::endl;
}

std::string Membrane::_getElementNameFromId(const size_t id)
{
    return _descriptor.assemblyName + "_Membrane_" + std::to_string(id);
}
} // namespace bioexplorer
