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

const int64_t SOMA_AS_PARENT = -1;
const size_t NB_MATERIALS_PER_ASTROCYTE = 2;

const size_t MATERIAL_OFFSET_SOMA = 0;
const size_t MATERIAL_OFFSET_DENDRITE = 1;

const double DEFAULT_SOMA_DISPLACEMENT = 2.0;
const double DEFAULT_SECTION_DISPLACEMENT = 5.0;

Astrocytes::Astrocytes(Scene& scene, const AstrocytesDetails& details)
    : Node(details.scale)
    , _details(details)
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
    const auto somas = connector.getAstrocytes(_details.sqlFilter);

    // Astrocytes
    size_t previousMaterialId = std::numeric_limits<size_t>::max();
    size_t baseMaterialId = 0;
    const uint64_t userData = 0;
    Vector3ui indexOffset;

    for (const auto& soma : somas)
    {
        const auto somaId = soma.first;
        const auto& somaPosition = soma.second.center;
        const auto somaRadius = soma.second.radius;

        PLUGIN_PROGRESS("Loading astrocytes", soma.first, somas.size());
        switch (_details.populationColorScheme)
        {
        case PopulationColorScheme::id:
            baseMaterialId = somaId * NB_MATERIALS_PER_ASTROCYTE;
            break;
        default:
            baseMaterialId = 0;
        }
        materialIds.insert(baseMaterialId);

        uint64_t somaGeometryIndex = 0;
        if (_details.loadSomas)
            somaGeometryIndex =
                _addSphere(useSdf, somaPosition, somaRadius, baseMaterialId,
                           NO_USER_DATA, *model, sdfMorphologyData, {},
                           DEFAULT_SOMA_DISPLACEMENT);

        const auto sections = connector.getAstrocyteSections(somaId);
        // End feet
        const auto endFeet =
            (_details.loadEndFeet ? connector.getAstrocyteEndFeetAreas(somaId)
                                  : EndFootMap());

        for (const auto& section : sections)
        {
            uint64_t geometryIndex = 0;
            const auto& points = section.second.points;

            if (section.second.parentId == SOMA_AS_PARENT)
            {
                // Section connected to the soma
                const auto& point = points[0];
                geometryIndex =
                    _addCone(useSdf, somaPosition, somaRadius,
                             somaPosition + Vector3d(point), point.w * 0.5,
                             baseMaterialId, NO_USER_DATA, *model,
                             sdfMorphologyData, {somaGeometryIndex},
                             DEFAULT_SOMA_DISPLACEMENT);
            }

            size_t sectionMaterialId = baseMaterialId;
            const auto sectionId = section.first;
            switch (_details.morphologyColorScheme)
            {
            case MorphologyColorScheme::section:
                sectionMaterialId = baseMaterialId + section.second.type;
                break;
            default:
                break;
            }
            materialIds.insert(sectionMaterialId);

            size_t step = 1;
            switch (_details.geometryQuality)
            {
            case GeometryQuality::low:
                step = points.size() - 2;
                break;
            default:
                break;
            }

            if (_details.loadDendrites)
            {
                uint64_t geometryIndex = 0;
                if (section.second.parentId == SOMA_AS_PARENT)
                {
                    // Section connected to the soma
                    const auto& point = points[0];
                    geometryIndex =
                        _addCone(useSdf, somaPosition, somaRadius * 0.5,
                                 somaPosition + Vector3d(point), point.w * 0.5,
                                 sectionMaterialId, userData, *model,
                                 sdfMorphologyData, {somaGeometryIndex},
                                 DEFAULT_SOMA_DISPLACEMENT);
                }

                for (uint64_t i = 0; i < points.size() - 1; i += step)
                {
                    const auto srcPoint = points[i];
                    const auto src = somaPosition + Vector3d(srcPoint);
                    const float srcRadius = srcPoint.w * 0.5;

                    // Ignore points that are too close the previous one
                    // (according to respective radii)
                    Vector4f dstPoint;
                    float dstRadius;
                    do
                    {
                        dstPoint = points[i + step];
                        dstRadius = dstPoint.w * 0.5;
                        ++i;
                    } while (length(Vector3f(dstPoint) - Vector3f(srcPoint)) <
                                 (srcRadius + dstRadius) &&
                             (i + step) < points.size() - 1);
                    --i;

                    const auto dst = somaPosition + Vector3d(dstPoint);
                    if (!useSdf)
                        geometryIndex =
                            _addSphere(useSdf, dst, dstRadius,
                                       sectionMaterialId, NO_USER_DATA, *model,
                                       sdfMorphologyData, {});

                    geometryIndex =
                        _addCone(useSdf, src, srcRadius, dst, dstRadius,
                                 sectionMaterialId, userData, *model,
                                 sdfMorphologyData, {geometryIndex},
                                 DEFAULT_SECTION_DISPLACEMENT);

                    _bounds.merge(srcPoint);
                }
            }

            if (_details.loadEndFeet)
            {
                const auto it = endFeet.find(sectionId);
                if (it != endFeet.end())
                {
                    const auto endFoot = (*it).second;
                    auto& tm = model->getTriangleMeshes()[sectionMaterialId];
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
            if (sectionMaterialId != previousMaterialId)
                indexOffset = Vector3ui();
            previousMaterialId = sectionMaterialId;
        }
    }

    _createMaterials(materialIds, *model);

    if (useSdf)
        _finalizeSDFGeometries(*model, sdfMorphologyData);

    ModelMetadata metadata = {
        {"Number of astrocytes", std::to_string(somas.size())}};

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
