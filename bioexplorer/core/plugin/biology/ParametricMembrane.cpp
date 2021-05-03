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

#include "ParametricMembrane.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Shapes.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>

namespace bioexplorer
{
namespace biology
{
using namespace common;

ParametricMembrane::ParametricMembrane(Scene &scene,
                                       const ParametricMembraneDetails &details)
    : Membrane(scene)
    , _details(details)
{
    if (_details.representation == ProteinRepresentation::contour)
    {
        switch (_details.shape)
        {
        case AssemblyShape::spherical:
            const std::string name{"ParametricMembrane"};
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
                             {_position, _details.assemblyParams[0]});
            auto modelDescriptor =
                std::make_shared<ModelDescriptor>(std::move(model), name);
            _scene.addModel(modelDescriptor);
            return;
        }
    }

    std::vector<std::string> proteinContents;
    proteinContents.push_back(details.content1);
    if (!details.content2.empty())
        proteinContents.push_back(details.content2);
    if (!details.content3.empty())
        proteinContents.push_back(details.content3);
    if (!details.content4.empty())
        proteinContents.push_back(details.content4);

    // Load proteins
    size_t i = 0;
    for (const auto &content : proteinContents)
    {
        ProteinDetails pd;
        pd.assemblyName = details.assemblyName;
        pd.name = _getElementNameFromId(i);
        pd.contents = content;
        pd.chainIds = details.chainIds;
        pd.recenter = details.recenter;
        pd.loadBonds = details.loadBonds;
        pd.randomSeed = details.randomSeed;
        pd.occurrences = 1;
        pd.rotation = details.rotation;
        pd.assemblyParams = details.assemblyParams;
        pd.atomRadiusMultiplier = details.atomRadiusMultiplier;
        pd.representation = details.representation;
        pd.loadNonPolymerChemicals = details.loadNonPolymerChemicals;
        pd.positionRandomizationType = details.positionRandomizationType;
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

ParametricMembrane::~ParametricMembrane()
{
    for (const auto &protein : _proteins)
        _scene.removeModel(protein.second->getModelDescriptor()->getModelID());
}

void ParametricMembrane::_processInstances()
{
    // Clipping planes
    const auto clipPlanes = getClippingPlanes(_scene);

    // Randomization
    srand(_details.randomSeed);

    const Quaterniond rotation = {_details.rotation[0], _details.rotation[1],
                                  _details.rotation[2], _details.rotation[3]};
    std::map<size_t, size_t> instanceCounts;
    for (size_t i = 0; i < _proteins.size(); ++i)
        instanceCounts[i] = 0;

    // Shape parameters
    const auto &params = _details.assemblyParams;
    const float size = (params.size() > 0 ? params[0] : 0.f);

    RandomizationDetails randInfo;
    randInfo.seed = _details.randomSeed;
    randInfo.randomizationType = _details.positionRandomizationType;
    randInfo.positionStrength = (params.size() > 2 ? params[2] : 0.f);
    randInfo.rotationStrength = (params.size() > 4 ? params[4] : 0.f);
    const float extraParameter = (params.size() > 5 ? params[5] : 0.f);

    // Shape instances
    const float offset = 2.f / _details.occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    for (uint64_t occurence = 0; occurence < _details.occurrences; ++occurence)
    {
        const size_t id = rand() % _proteins.size();
        auto protein = _proteins[_getElementNameFromId(id)];
        auto md = protein->getModelDescriptor();

        const auto &model = md->getModel();
        const auto &bounds = model.getBounds();
        const Vector3f &center = bounds.getCenter();

        randInfo.positionSeed =
            (params.size() >= 2 ? (params[1] == 0 ? 0 : params[1] + occurence)
                                : 0);
        randInfo.rotationSeed =
            (params.size() >= 4 ? (params[3] == 0 ? 0 : params[3] + occurence)
                                : 0);

        Transformation transformation;
        switch (_details.shape)
        {
        case AssemblyShape::spherical:
        {
            transformation =
                getSphericalPosition(Vector3f(), size, occurence,
                                     _details.occurrences, randInfo);
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
                                            _details.occurrences, randInfo);
            break;
        }
        case AssemblyShape::bezier:
        {
            if ((params.size() - 5) % 3 != 0)
                PLUGIN_THROW(
                    "Invalid number of floats in assembly extra parameters");
            Vector3fs points;
            for (uint32_t i = 5; i < params.size(); i += 3)
                points.push_back(
                    Vector3f(params[i], params[i + 1], params[i + 2]));
            transformation = getBezierPosition(points, size,
                                               float(occurence) /
                                                   float(_details.occurrences));
            break;
        }
        case AssemblyShape::spherical_to_planar:
        {
            transformation =
                getSphericalToPlanarPosition(Vector3f(), size, occurence,
                                             _details.occurrences, randInfo,
                                             extraParameter);
            break;
        }
        default:
            transformation = getPlanarPosition(Vector3f(), size, randInfo);
            break;
        }

        Vector3f translation = Vector3f(
            _rotation * (transformation.getTranslation() - Vector3d(center)));

        // Clipping planes
        if (isClipped(translation, clipPlanes))
            continue;

        // Final transformation
        translation += _position;

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

std::string ParametricMembrane::_getElementNameFromId(const size_t id)
{
    return _details.assemblyName + "_Membrane_" + std::to_string(id);
}
} // namespace biology
} // namespace bioexplorer
