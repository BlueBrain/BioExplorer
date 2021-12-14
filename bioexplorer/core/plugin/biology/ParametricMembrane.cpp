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
                                       const Vector3f &assemblyPosition,
                                       const Quaterniond &assemblyRotation,
                                       const Vector4fs &clippingPlanes,
                                       const ParametricMembraneDetails &details,
                                       const ModelDescriptors &modelDescriptors)
    : Membrane(scene, assemblyPosition, assemblyRotation, clippingPlanes)
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
                             {_assemblyPosition, _details.assemblyParams[0]});
            auto modelDescriptor =
                std::make_shared<ModelDescriptor>(std::move(model), name);
            _scene.addModel(modelDescriptor);
            return;
        }
    }

    std::vector<std::string> lipidContents =
        split(details.lipidContents, PDB_CONTENTS_DELIMITER);

    // Load lipids
    size_t i = 0;
    for (const auto &content : lipidContents)
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

        ProteinPtr lipid(new Protein(_scene, pd));
        auto modelDescriptor = lipid->getModelDescriptor();
        _lipids[pd.name] = std::move(lipid);
        ++i;
    }

    // Assemble lipids
    _processInstances(modelDescriptors);

    // Add models to the scene
    for (size_t i = 0; i < lipidContents.size(); ++i)
        _scene.addModel(
            _lipids[_getElementNameFromId(i)]->getModelDescriptor());
}

void ParametricMembrane::_processInstances(
    const ModelDescriptors &modelDescriptors)
{
    // Randomization
    srand(_details.randomSeed);

    const auto rotation = floatsToQuaterniond(_details.rotation);
    std::map<size_t, size_t> instanceCounts;
    for (size_t i = 0; i < _lipids.size(); ++i)
        instanceCounts[i] = 0;

    // Shape parameters
    const auto &params = _details.assemblyParams;
    const Vector3f size =
        (params.size() > 0 ? Vector3f(params[PARAMS_OFFSET_DIMENSION_1],
                                      params[PARAMS_OFFSET_DIMENSION_2],
                                      params[PARAMS_OFFSET_DIMENSION_3])
                           : Vector3f(0.f));
    const auto extraParameter =
        (params.size() > PARAMS_OFFSET_EXTRA ? params[PARAMS_OFFSET_EXTRA]
                                             : 0.f);
    auto randInfo =
        floatsToRandomizationDetails(params, _details.randomSeed,
                                     _details.positionRandomizationType);

    // Shape instances
    const float offset = 2.f / _details.occurrences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    for (uint64_t occurence = 0; occurence < _details.occurrences; ++occurence)
    {
        const size_t id = rand() % _lipids.size();
        auto lipid = _lipids[_getElementNameFromId(id)];
        auto md = lipid->getModelDescriptor();

        const auto &model = md->getModel();
        const auto &bounds = model.getBounds();
        const Vector3f &center = bounds.getCenter();

        randInfo.positionSeed =
            (params.size() > PARAMS_OFFSET_POSITION_SEED
                 ? (params[PARAMS_OFFSET_POSITION_SEED] == 0
                        ? 0
                        : params[PARAMS_OFFSET_POSITION_SEED] + occurence)
                 : 0);
        randInfo.rotationSeed =
            (params.size() > PARAMS_OFFSET_ROTATION_SEED
                 ? (params[PARAMS_OFFSET_ROTATION_SEED] == 0
                        ? 0
                        : params[PARAMS_OFFSET_ROTATION_SEED] + occurence)
                 : 0);

        Transformation transformation;
        switch (_details.shape)
        {
        case AssemblyShape::spherical:
        {
            transformation =
                getSphericalPosition(Vector3f(), size.x, occurence,
                                     _details.occurrences, randInfo);
            break;
        }
        case AssemblyShape::sinusoidal:
        {
            transformation =
                getSinosoidalPosition(Vector3f(), Vector2f(size.x, size.z),
                                      extraParameter, occurence, randInfo);
            break;
        }
        case AssemblyShape::cubic:
        {
            transformation = getCubicPosition(Vector3f(), size, randInfo);
            break;
        }
        case AssemblyShape::fan:
        {
            transformation = getFanPosition(Vector3f(), size.x, occurence,
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
            transformation = getBezierPosition(points, size.x,
                                               float(occurence) /
                                                   float(_details.occurrences));
            break;
        }
        case AssemblyShape::spherical_to_planar:
        {
            transformation =
                getSphericalToPlanarPosition(Vector3f(), size.x, occurence,
                                             _details.occurrences, randInfo,
                                             extraParameter);
            break;
        }
        default:
            transformation = getPlanarPosition(Vector3f(), size.x, randInfo);
            break;
        }

        // Clipping planes
        if (isClipped(transformation.getTranslation(), _clippingPlanes))
            continue;

        const Vector3f translation =
            _assemblyPosition +
            Vector3f(_assemblyRotation *
                     (transformation.getTranslation() - Vector3d(center)));

        // Collision with trans-membrane proteins
        bool collision = false;
        for (const auto &modelDescriptor : modelDescriptors)
        {
            const auto &instances = modelDescriptor->getInstances();
            const auto &instanceSize =
                modelDescriptor->getModel().getBounds().getSize();
            for (const auto &instance : instances)
            {
                const auto &tf = instance.getTransformation();
                const Vector3f &t = tf.getTranslation();
                if (length(translation - t) < instanceSize.x / 2.0)
                {
                    collision = true;
                    break;
                }
            }
        }
        if (collision)
            continue;

        // Final transformation
        Transformation finalTransformation;
        finalTransformation.setTranslation(translation);
        finalTransformation.setRotation(
            _assemblyRotation * transformation.getRotation() * rotation);

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

bool ParametricMembrane::isInside(const Vector3f &point) const
{
    PLUGIN_THROW("ParametricMembrane::isInside is Not implemented");
}
} // namespace biology
} // namespace bioexplorer
