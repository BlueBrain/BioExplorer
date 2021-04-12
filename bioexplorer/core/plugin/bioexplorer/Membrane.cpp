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
                   const Vector3f &position, const Quaterniond &rotation,
                   const Vector4fs &clippingPlanes,
                   const OccupiedDirections &occupiedDirections)
    : _scene(scene)
    , _position(position)
    , _rotation(rotation)
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
        pd.rotation = descriptor.rotation;
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

    const Quaterniond rotation = {_descriptor.rotation[0],
                                  _descriptor.rotation[1],
                                  _descriptor.rotation[2],
                                  _descriptor.rotation[3]};
    std::map<size_t, size_t> instanceCounts;
    for (size_t i = 0; i < _proteins.size(); ++i)
        instanceCounts[i] = 0;

    // Shape parameters
    const auto &params = _descriptor.assemblyParams;
    if (params.size() < 6)
        PLUGIN_THROW(std::runtime_error("Invalid number of shape parameters"));

    const float size = params[0];

    RandomizationInformation randInfo;
    randInfo.seed = _descriptor.randomSeed;
    randInfo.randomizationType = _descriptor.positionRandomizationType;
    randInfo.positionStrength = params[2];
    randInfo.rotationStrength = params[4];
    const float extraParameter = params[5];

    // Shape instances
    const float offset = 2.f / _descriptor.occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    for (uint64_t occurence = 0; occurence < _descriptor.occurrences;
         ++occurence)
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

        randInfo.positionSeed = (params[1] == 0 ? 0 : params[1] + occurence);
        randInfo.rotationSeed = (params[3] == 0 ? 0 : params[3] + occurence);

        Transformation transformation;
        switch (_descriptor.shape)
        {
        case AssemblyShape::spherical:
        {
            transformation =
                getSphericalPosition(Vector3f(), size, occurence,
                                     _descriptor.occurrences, randInfo);
            break;
        }
        case AssemblyShape::sinusoidal:
        {
            transformation =
                getSinosoidalPosition(Vector3f(), size, extraParameter,
                                      occurence, randInfo);
            break;
        }
        case AssemblyShape::cubic:
        {
            transformation = getCubicPosition(Vector3f(), size, randInfo);
            break;
        }
        case AssemblyShape::fan:
        {
            transformation = getFanPosition(Vector3f(), size, occurence,
                                            _descriptor.occurrences, randInfo);
            break;
        }
        case AssemblyShape::bezier:
        {
            if ((params.size() - 5) % 3 != 0)
                PLUGIN_THROW(std::runtime_error(
                    "Invalid number of floats in assembly extra parameters"));
            Vector3fs points;
            for (uint32_t i = 5; i < params.size(); i += 3)
                points.push_back(
                    Vector3f(params[i], params[i + 1], params[i + 2]));
            transformation =
                getBezierPosition(points, size,
                                  float(occurence) /
                                      float(_descriptor.occurrences));
            break;
        }
        case AssemblyShape::spherical_to_planar:
        {
            transformation =
                getSphericalToPlanarPosition(Vector3f(), size, occurence,
                                             _descriptor.occurrences, randInfo,
                                             extraParameter);
            break;
        }
        default:
            transformation = getPlanarPosition(Vector3f(), size, randInfo);
            break;
        }

#if 0 // TO REMOVE ?
      // Remove membrane where proteins are. This is currently done
      // according to the vector rotation
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
#endif

        // Final transformation
        const Vector3f translation =
            _position + Vector3f(_rotation * (transformation.getTranslation() -
                                              Vector3d(center)));

        if (isClipped(translation, _clippingPlanes))
            continue;

        Transformation finalTransformation;
        finalTransformation.setTranslation(translation);
        finalTransformation.setRotation(
            _rotation * transformation.getRotation() * rotation);

        if (instanceCounts[id] == 0)
            md->setTransformation(finalTransformation);
        const ModelInstance instance(true, false, finalTransformation);
        md->addInstance(instance);

        instanceCounts[id] = instanceCounts[id] + 1;
    }
}

std::string Membrane::_getElementNameFromId(const size_t id)
{
    return _descriptor.assemblyName + "_Membrane_" + std::to_string(id);
}
} // namespace bioexplorer
