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

#include "MeshCircuitLoader.h"

#include <common/Logs.h>

using namespace core;

namespace sonataexplorer
{
namespace neuroscience
{
using namespace common;

namespace neuron
{
const std::string LOADER_NAME = "Sonata circuit with meshes";
const double DEFAULT_RADIUS_MULTIPLIER = 2.0;

MeshCircuitLoader::MeshCircuitLoader(Scene &scene, const ApplicationParameters &applicationParameters,
                                     PropertyMap &&loaderParams)
    : AbstractCircuitLoader(scene, applicationParameters, std::move(loaderParams))
{
    _fixedDefaults.setProperty({PROP_PRESYNAPTIC_NEURON_GID.name, std::string("")});
    _fixedDefaults.setProperty({PROP_POSTSYNAPTIC_NEURON_GID.name, std::string("")});
    _fixedDefaults.setProperty({PROP_REPORT_TYPE.name, enumToString(ReportType::voltages_from_file)});
    _fixedDefaults.setProperty({PROP_CIRCUIT_COLOR_SCHEME.name, enumToString(CircuitColorScheme::by_id)});
    _fixedDefaults.setProperty({PROP_RADIUS_MULTIPLIER.name, DEFAULT_RADIUS_MULTIPLIER});
    _fixedDefaults.setProperty({PROP_RADIUS_CORRECTION.name, 0.0});
    _fixedDefaults.setProperty({PROP_USE_SDF_SOMA.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_BRANCHES.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_NUCLEUS.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_MITOCHONDRIA.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_MYELIN_STEATH.name, false});
    _fixedDefaults.setProperty({PROP_DAMPEN_BRANCH_THICKNESS_CHANGERATE.name, false});
    _fixedDefaults.setProperty({PROP_USER_DATA_TYPE.name, enumToString(UserDataType::simulation_offset)});
    _fixedDefaults.setProperty({PROP_ASSET_COLOR_SCHEME.name, enumToString(AssetColorScheme::none)});
    _fixedDefaults.setProperty({PROP_ASSET_QUALITY.name, enumToString(AssetQuality::high)});
    _fixedDefaults.setProperty({PROP_MORPHOLOGY_MAX_DISTANCE_TO_SOMA.name, std::numeric_limits<double>::max()});
    _fixedDefaults.setProperty({PROP_CELL_CLIPPING.name, false});
    _fixedDefaults.setProperty({PROP_AREAS_OF_INTEREST.name, 0});
    _fixedDefaults.setProperty({PROP_LOAD_AFFERENT_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_LOAD_EFFERENT_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_INTERNALS.name, false});
    _fixedDefaults.setProperty({PROP_EXTERNALS.name, false});
}

ModelDescriptorPtr MeshCircuitLoader::importFromStorage(const std::string &path, const LoaderProgress &callback,
                                                        const PropertyMap &properties) const
{
    PLUGIN_INFO("Loading circuit from " << path);
    callback.updateProgress("Loading circuit ...", 0);
    PropertyMap props = _defaults;
    props.merge(_fixedDefaults);
    props.merge(properties);
    return importCircuit(path, props, callback);
}

std::string MeshCircuitLoader::getName() const
{
    return LOADER_NAME;
}

PropertyMap MeshCircuitLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(PROP_DENSITY);
    pm.setProperty(PROP_REPORT);
    pm.setProperty(PROP_SYNCHRONOUS_MODE);
    pm.setProperty(PROP_VOLTAGE_SCALING);
    pm.setProperty(PROP_TARGETS);
    pm.setProperty(PROP_GIDS);
    pm.setProperty(PROP_RANDOM_SEED);
    pm.setProperty(PROP_MESH_FOLDER);
    pm.setProperty(PROP_MESH_FILENAME_PATTERN);
    pm.setProperty(PROP_MESH_TRANSFORMATION);
    pm.setProperty(PROP_SECTION_TYPE_SOMA);
    pm.setProperty(PROP_SECTION_TYPE_AXON);
    pm.setProperty(PROP_SECTION_TYPE_DENDRITE);
    pm.setProperty(PROP_SECTION_TYPE_APICAL_DENDRITE);
    return pm;
}
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
