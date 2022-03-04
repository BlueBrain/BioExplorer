/* Copyright (c) 2018-2021, EPFL/Blue Brain Project
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

#include "Astrocytes.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>

#include <plugin/io/db/DBConnector.h>

#include <brayns/common/Timer.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace morphology
{
using namespace common;
using namespace io;
using namespace db;

const size_t NB_MATERIALS_PER_ASTROCYTE = 4;

Astrocytes::Astrocytes(Scene& scene, const AstrocytesDetails& details)
    : _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Astrocytes loaded");
}

void Astrocytes::_buildModel()
{
    auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    std::set<uint64_t> materialIds;
    SDFMorphologyData sdfMorphologyData;
    const auto useSdf = _details.useSdf;
    const auto somas =
        connector.getNodes(_details.astrocyteIds, _details.sqlFilter);

    // Astrocytes
    size_t previousMaterialId = std::numeric_limits<size_t>::max();
    size_t materialId = 0;
    const uint64_t userData = 0;
    Vector3ui indexOffset;

    for (const auto& soma : somas)
    {
        const auto somaId = soma.first;
        const auto& somaCenter = soma.second.center;
        const auto somaRadius = soma.second.radius;

        PLUGIN_PROGRESS("Loading astrocytes", soma.first, somas.size());
        switch (_details.populationColorScheme)
        {
        case PopulationColorScheme::id:
            materialId = somaId * NB_MATERIALS_PER_ASTROCYTE;
            break;
        default:
            materialId = 0;
        }
        materialIds.insert(materialId);

        if (_details.loadSomas)
        {
            if (useSdf)
                _addSDFGeometry(sdfMorphologyData,
                                createSDFSphere(somaCenter, somaRadius,
                                                userData),
                                {}, materialId, -1);
            else
                model->addSphere(materialId,
                                 {somaCenter, static_cast<float>(somaRadius),
                                  userData});
        }

        if (!_details.loadDendrites && !_details.loadEndFeet)
            continue;

        const auto sections = connector.getAstrocyteSections(somaId);
        // End feet
        const auto endFeet =
            (_details.loadEndFeet ? connector.getAstrocyteEndFeetAreas(somaId)
                                  : EndFootMap());

        for (const auto& section : sections)
        {
            const auto sectionId = section.first;
            switch (_details.morphologyColorScheme)
            {
            case MorphologyColorScheme::section:
                materialId += section.second.type;
                break;
            default:
                break;
            }
            materialIds.insert(materialId);

            const auto& points = section.second.points;
            size_t step = 1;
            switch (_details.geometryQuality)
            {
            case GeometryQuality::low:
                step = points.size() - 1;
                break;
            default:
                break;
            }

            if (_details.loadDendrites)
            {
                if (section.second.parentId == -1)
                {
                    const auto& point = points[0];
                    _addStepConeGeometry(useSdf, somaCenter, somaRadius,
                                         somaCenter + Vector3d(point), point.w,
                                         materialId, userData, *model,
                                         sdfMorphologyData, sectionId);
                }

                for (uint64_t i = 0; i < points.size() - 2; i += step)
                {
                    const auto& srcPoint = points[i];
                    const auto src = somaCenter + Vector3d(srcPoint);
                    const float srcRadius = srcPoint.w * 0.5;

                    _bounds.merge(srcPoint);

                    const auto& dstPoint = points[i + step];
                    const auto dst = somaCenter + Vector3d(dstPoint);
                    const float dstRadius = dstPoint.w * 0.5;

                    if (useSdf)
                        _addSDFGeometry(sdfMorphologyData,
                                        createSDFSphere(dst, dstRadius,
                                                        userData),
                                        {}, materialId, sectionId);
                    else
                        model->addSphere(materialId,
                                         {dst, dstRadius, userData});

                    _addStepConeGeometry(useSdf, src, srcRadius, dst, dstRadius,
                                         materialId, userData, *model,
                                         sdfMorphologyData, sectionId);
                }
            }

            if (_details.loadEndFeet)
            {
                const auto it = endFeet.find(sectionId);
                if (it != endFeet.end())
                {
                    const auto endFoot = (*it).second;
                    auto& tm = model->getTriangleMeshes()[materialId];
                    for (const auto& vertex : endFoot.vertices)
                    {
                        _bounds.merge(vertex);
                        tm.vertices.push_back(vertex);
                    }
                    for (const auto& index : endFoot.indices)
                        tm.indices.push_back(index + indexOffset);

                    const auto nbVertices = endFoot.vertices.size();
                    indexOffset +=
                        Vector3ui(nbVertices, nbVertices, nbVertices);
                }
            }
            if (materialId != previousMaterialId)
                indexOffset = Vector3ui();
            previousMaterialId = materialId;
        }
    }

    for (const auto materialId : materialIds)
    {
        Vector3f color{1.f, 1.f, 1.f};
        auto nodeMaterial =
            model->createMaterial(materialId, std::to_string(materialId));
        nodeMaterial->setDiffuseColor(color);
        nodeMaterial->setSpecularColor(color);
        nodeMaterial->setSpecularExponent(100.f);
        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_CAST_USER_DATA, false});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, 0});
        nodeMaterial->updateProperties(props);
    }

    if (useSdf)
        _finalizeSDFGeometries(*model, sdfMorphologyData);

    ModelMetadata metadata = {
        {"Number of astrocytes", std::to_string(_details.astrocyteIds.size())}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW("Astrocytes model could not be created");
}
} // namespace morphology
} // namespace bioexplorer