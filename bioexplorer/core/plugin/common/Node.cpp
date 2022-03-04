/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "Node.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/UniqueId.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace common
{
// From http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x - y) <=
               std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
           // unless the result is subnormal
           || std::abs(x - y) < std::numeric_limits<T>::min();
}

Node::Node()
{
    // Unique ID
    _uuid = UniqueId::get();
}

const ModelDescriptorPtr Node::getModelDescriptor() const
{
    return _modelDescriptor;
}

void Node::_setMaterialExtraAttributes()
{
    auto materials = _modelDescriptor->getModel().getMaterials();
    for (auto& material : materials)
    {
        brayns::PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE,
                           static_cast<int>(MaterialChameleonMode::receiver)});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, static_cast<int>(_uuid)});
        material.second->updateProperties(props);
    }
}

// TODO: Generalise SDF for any type of asset
size_t Node::_addSDFGeometry(SDFMorphologyData& sdfMorphologyData,
                             const SDFGeometry& geometry,
                             const std::set<size_t>& neighbours,
                             const size_t materialId, const int section)
{
    const size_t idx = sdfMorphologyData.geometries.size();
    sdfMorphologyData.geometries.push_back(geometry);
    sdfMorphologyData.neighbours.push_back(neighbours);
    sdfMorphologyData.materials.push_back(materialId);
    sdfMorphologyData.geometrySection[idx] = section;
    sdfMorphologyData.sectionGeometries[section].push_back(idx);
    return idx;
}

void Node::_addStepSphereGeometry(const bool useSDF, const Vector3d& position,
                                  const double radius, const size_t materialId,
                                  const uint64_t userData, Model& model,
                                  SDFMorphologyData& sdfMorphologyData,
                                  const uint32_t sdfGroupId,
                                  const Vector3f& displacementParams)
{
    if (useSDF)
        _addSDFGeometry(sdfMorphologyData,
                        createSDFSphere(position, radius, userData,
                                        displacementParams),
                        {}, materialId, sdfGroupId);
    else
        model.addSphere(materialId,
                        {position, static_cast<float>(radius), userData});
}

void Node::_addStepConeGeometry(const bool useSDF, const Vector3d& position,
                                const double radius, const Vector3d& target,
                                const double previousRadius,
                                const size_t materialId,
                                const uint64_t userData, Model& model,
                                SDFMorphologyData& sdfMorphologyData,
                                const uint32_t sdfGroupId,
                                const Vector3f& displacementParams)
{
    if (useSDF)
    {
        const auto geom =
            (almost_equal(radius, previousRadius, 100000))
                ? createSDFPill(position, target, radius, userData,
                                displacementParams)
                : createSDFConePill(position, target, radius, previousRadius,
                                    userData, displacementParams);
        _addSDFGeometry(sdfMorphologyData, geom, {}, materialId, sdfGroupId);
    }
    else if (almost_equal(radius, previousRadius, 100000))
        model.addCylinder(materialId, {position, target,
                                       static_cast<float>(radius), userData});
    else
        model.addCone(materialId,
                      {position, target, static_cast<float>(radius),
                       static_cast<float>(previousRadius), userData});
}

void Node::_finalizeSDFGeometries(Model& model,
                                  SDFMorphologyData& sdfMorphologyData)
{
    const size_t numGeoms = sdfMorphologyData.geometries.size();
    sdfMorphologyData.localToGlobalIdx.resize(numGeoms, 0);

    // Extend neighbours to make sure smoothing is applied on all
    // closely connected geometries
    for (size_t rep = 0; rep < 4; rep++)
    {
        const size_t numNeighs = sdfMorphologyData.neighbours.size();
        auto neighsCopy = sdfMorphologyData.neighbours;
        for (size_t i = 0; i < numNeighs; i++)
        {
            for (size_t j : sdfMorphologyData.neighbours[i])
            {
                for (size_t newNei : sdfMorphologyData.neighbours[j])
                {
                    neighsCopy[i].insert(newNei);
                    neighsCopy[newNei].insert(i);
                }
            }
        }
        sdfMorphologyData.neighbours = neighsCopy;
    }

    for (size_t i = 0; i < numGeoms; i++)
    {
        // Convert neighbours from set to vector and erase itself from its
        // neighbours
        std::vector<size_t> neighbours;
        const auto& neighSet = sdfMorphologyData.neighbours[i];
        std::copy(neighSet.begin(), neighSet.end(),
                  std::back_inserter(neighbours));
        neighbours.erase(std::remove_if(neighbours.begin(), neighbours.end(),
                                        [i](size_t elem) { return elem == i; }),
                         neighbours.end());

        model.addSDFGeometry(sdfMorphologyData.materials[i],
                             sdfMorphologyData.geometries[i], neighbours);
    }
}

} // namespace common
} // namespace bioexplorer
