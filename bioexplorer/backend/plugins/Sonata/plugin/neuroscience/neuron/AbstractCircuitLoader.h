/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <plugin/api/SonataExplorerParams.h>
#include <plugin/neuroscience/common/Types.h>

#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>

#include <set>
#include <vector>

namespace servus
{
class URI;
}

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
/**
 * Load circuit from BlueConfig or CircuitConfig file, including simulation.
 */
class AbstractCircuitLoader : public core::Loader
{
public:
    AbstractCircuitLoader(core::Scene &scene, const core::ApplicationParameters &applicationParameters,
                          core::PropertyMap &&loaderParams);

    core::PropertyMap getProperties() const final;

    strings getSupportedStorage() const;

    bool isSupported(const std::string &filename, const std::string &extension) const;

    core::ModelDescriptorPtr importFromBlob(core::Blob &&blob, const core::LoaderProgress &callback,
                                            const core::PropertyMap &properties) const;

    /**
     * @brief Imports morphology from a circuit for the given target name
     * @param circuitConfig URI of the Circuit Config file
     * @return ModelDescriptor if the circuit is successfully loaded, nullptr if
     * the circuit contains no cells.
     */
    core::ModelDescriptorPtr importCircuit(const std::string &circuitConfig, const core::PropertyMap &properties,
                                           const core::LoaderProgress &callback) const;

    /**
     * @brief _populateLayerIds populates the neuron layer IDs. This is
     * currently only supported for the MVD2 format.
     * @param blueConfig Configuration of the circuit
     * @param gids GIDs of the neurons
     */
    size_ts _populateLayerIds(const core::PropertyMap &props, const brion::BlueConfig &blueConfig,
                              const brain::GIDSet &gids) const;

    static void setSimulationTransferFunction(core::TransferFunction &tf, const float finalOpacity = 1.f);

protected:
    const core::ApplicationParameters &_applicationParameters;
    core::PropertyMap _defaults;
    core::PropertyMap _fixedDefaults;

private:
    std::vector<std::string> _getTargetsAsStrings(const std::string &targets) const;

    brain::GIDSet _getGids(const core::PropertyMap &properties, const brion::BlueConfig &blueConfiguration,
                           const brain::Circuit &circuit, common::GIDOffsets &targetGIDOffsets) const;

    std::string _getMeshFilenameFromGID(const core::PropertyMap &props, const uint64_t gid) const;

    float _importMorphologies(const core::PropertyMap &props, const brain::Circuit &circuit, core::Model &model,
                              const brain::GIDSet &gids, const common::Matrix4fs &transformations,
                              const common::GIDOffsets &targetGIDOffsets,
                              common::CompartmentReportPtr compartmentReport, const size_ts &layerIds,
                              const size_ts &morphologyTypes, const size_ts &electrophysiologyTypes,
                              const core::LoaderProgress &callback, const size_t materialId = core::NO_MATERIAL) const;

    /**
     * @brief _getMaterialFromSectionType return a material determined by the
     * --color-scheme geometry parameter
     * @param index Index of the element to which the material will attached
     * @param material Material that is forced in case geometry parameters
     * do not apply
     * @param sectionType Section type of the geometry to which the material
     * will be applied
     * @return Material ID determined by the geometry parameters
     */
    size_t _getMaterialFromCircuitAttributes(const core::PropertyMap &props, const uint64_t index,
                                             const size_t material, const common::GIDOffsets &targetGIDOffsets,
                                             const size_ts &layerIds, const size_ts &morphologyTypes,
                                             const size_ts &electrophysiologyTypes,
                                             const bool forSimulationModel) const;

    void _importMeshes(const core::PropertyMap &props, core::Model &model, const brain::GIDSet &gids,
                       const common::Matrix4fs &transformations, const common::GIDOffsets &targetGIDOffsets,
                       const size_ts &layerIds, const size_ts &morphologyTypes, const size_ts &electrophysiologyTypes,
                       const core::LoaderProgress &callback) const;

    common::CompartmentReportPtr _attachSimulationHandler(const core::PropertyMap &properties,
                                                          const brion::BlueConfig &blueConfiguration,
                                                          core::Model &model, const common::ReportType &reportType,
                                                          brain::GIDSet &gids) const;

    void _filterGIDsWithClippingPlanes(brain::GIDSet &gids, common::Matrix4fs &transformations) const;

    void _filterGIDsWithAreasOfInterest(const uint16_t areasOfInterest, brain::GIDSet &gids,
                                        common::Matrix4fs &transformations) const;

    bool _isClipped(const core::Vector3f &position) const;

    void _setDefaultCircuitColorMap(core::Model &model) const;
};
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
