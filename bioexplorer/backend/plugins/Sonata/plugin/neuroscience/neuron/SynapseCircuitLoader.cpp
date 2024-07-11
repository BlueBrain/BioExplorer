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

#include "SynapseCircuitLoader.h"

#include <common/Logs.h>

using namespace core;

namespace sonataexplorer
{
namespace neuroscience
{
using namespace common;

namespace neuron
{
const std::string LOADER_NAME = "Sonata synapses";

SynapseCircuitLoader::SynapseCircuitLoader(Scene &scene, const ApplicationParameters &applicationParameters,
                                           PropertyMap &&loaderParams)
    : AbstractCircuitLoader(scene, applicationParameters, std::move(loaderParams))
{
    _fixedDefaults.setProperty({PROP_REPORT.name, std::string("")});
    _fixedDefaults.setProperty({PROP_PRESYNAPTIC_NEURON_GID.name, std::string("")});
    _fixedDefaults.setProperty({PROP_POSTSYNAPTIC_NEURON_GID.name, std::string("")});
    _fixedDefaults.setProperty({PROP_REPORT_TYPE.name, enumToString(ReportType::undefined)});
    _fixedDefaults.setProperty({PROP_RADIUS_CORRECTION.name, 0.0});
    _fixedDefaults.setProperty({PROP_DAMPEN_BRANCH_THICKNESS_CHANGERATE.name, true});
    _fixedDefaults.setProperty({PROP_USER_DATA_TYPE.name, enumToString(UserDataType::undefined)});
    _fixedDefaults.setProperty({PROP_MORPHOLOGY_MAX_DISTANCE_TO_SOMA.name, std::numeric_limits<double>::max()});
    _fixedDefaults.setProperty({PROP_MESH_FOLDER.name, std::string("")});
    _fixedDefaults.setProperty({PROP_MESH_FILENAME_PATTERN.name, std::string("")});
    _fixedDefaults.setProperty({PROP_MESH_TRANSFORMATION.name, false});
    _fixedDefaults.setProperty({PROP_CELL_CLIPPING.name, false});
    _fixedDefaults.setProperty({PROP_AREAS_OF_INTEREST.name, 0});
    _fixedDefaults.setProperty({PROP_USE_SDF_NUCLEUS.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_MITOCHONDRIA.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_MYELIN_STEATH.name, false});
    _fixedDefaults.setProperty(PROP_POSITION);
    _fixedDefaults.setProperty(PROP_ROTATION);
}

ModelDescriptorPtr SynapseCircuitLoader::importFromStorage(const std::string &path, const LoaderProgress &callback,
                                                           const PropertyMap &properties) const
{
    PLUGIN_INFO("Loading circuit from " << path);
    callback.updateProgress("Loading circuit ...", 0);
    PropertyMap props = _defaults;
    props.merge(_fixedDefaults);
    props.merge(properties);
    return importCircuit(path, props, callback);
}

std::string SynapseCircuitLoader::getName() const
{
    return LOADER_NAME;
}

PropertyMap SynapseCircuitLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(PROP_DENSITY);
    pm.setProperty(PROP_TARGETS);
    pm.setProperty(PROP_GIDS);
    pm.setProperty(PROP_RADIUS_MULTIPLIER);
    pm.setProperty(PROP_RANDOM_SEED);
    pm.setProperty(PROP_CIRCUIT_COLOR_SCHEME);
    pm.setProperty(PROP_SECTION_TYPE_SOMA);
    pm.setProperty(PROP_SECTION_TYPE_AXON);
    pm.setProperty(PROP_SECTION_TYPE_DENDRITE);
    pm.setProperty(PROP_SECTION_TYPE_APICAL_DENDRITE);
    pm.setProperty(PROP_USE_SDF_SOMA);
    pm.setProperty(PROP_USE_SDF_BRANCHES);
    pm.setProperty(PROP_ASSET_COLOR_SCHEME);
    pm.setProperty(PROP_ASSET_QUALITY);
    pm.setProperty(PROP_LOAD_AFFERENT_SYNAPSES);
    pm.setProperty(PROP_LOAD_EFFERENT_SYNAPSES);
    pm.setProperty(PROP_INTERNALS);
    pm.setProperty(PROP_EXTERNALS);
    pm.setProperty(PROP_POSITION);
    pm.setProperty(PROP_ROTATION);
    return pm;
}
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
