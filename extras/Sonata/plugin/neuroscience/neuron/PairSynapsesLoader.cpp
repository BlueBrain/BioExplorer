/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "PairSynapsesLoader.h"

#include <common/Logs.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
const std::string LOADER_NAME = "Pair synapses";

PairSynapsesLoader::PairSynapsesLoader(
    Scene &scene, const ApplicationParameters &applicationParameters,
    PropertyMap &&loaderParams)
    : AbstractCircuitLoader(scene, applicationParameters,
                            std::move(loaderParams))
{
    _fixedDefaults.setProperty({PROP_DENSITY.name, 1.0});
    _fixedDefaults.setProperty({PROP_RANDOM_SEED.name, 0.0});
    _fixedDefaults.setProperty({PROP_REPORT.name, std::string("")});
    _fixedDefaults.setProperty({PROP_TARGETS.name, std::string("")});
    _fixedDefaults.setProperty({PROP_GIDS.name, std::string("")});
    _fixedDefaults.setProperty(
        {PROP_REPORT_TYPE.name, enumToString(ReportType::undefined)});
    _fixedDefaults.setProperty({PROP_RADIUS_CORRECTION.name, 0.0});
    _fixedDefaults.setProperty({PROP_CIRCUIT_COLOR_SCHEME.name,
                                enumToString(CircuitColorScheme::by_id)});
    _fixedDefaults.setProperty(
        {PROP_DAMPEN_BRANCH_THICKNESS_CHANGERATE.name, true});
    _fixedDefaults.setProperty(
        {PROP_USER_DATA_TYPE.name, enumToString(UserDataType::undefined)});
    _fixedDefaults.setProperty({PROP_MORPHOLOGY_MAX_DISTANCE_TO_SOMA.name,
                                std::numeric_limits<double>::max()});
    _fixedDefaults.setProperty({PROP_MESH_FOLDER.name, std::string("")});
    _fixedDefaults.setProperty(
        {PROP_MESH_FILENAME_PATTERN.name, std::string("")});
    _fixedDefaults.setProperty({PROP_MESH_TRANSFORMATION.name, false});
    _fixedDefaults.setProperty({PROP_CELL_CLIPPING.name, false});
    _fixedDefaults.setProperty({PROP_AREAS_OF_INTEREST.name, 0});
    _fixedDefaults.setProperty({PROP_LOAD_AFFERENT_SYNAPSES.name, true});
    _fixedDefaults.setProperty({PROP_LOAD_EFFERENT_SYNAPSES.name, true});
    _fixedDefaults.setProperty({PROP_INTERNALS.name, false});
    _fixedDefaults.setProperty({PROP_EXTERNALS.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_NUCLEUS.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_MITOCHONDRIA.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_USE_SDF_MYELIN_STEATH.name, false});
}

ModelDescriptorPtr PairSynapsesLoader::importFromFile(
    const std::string &filename, const LoaderProgress &callback,
    const PropertyMap &properties) const
{
    PLUGIN_INFO("Loading circuit from " << filename);
    callback.updateProgress("Loading circuit ...", 0);
    PropertyMap props = _defaults;
    props.merge(_fixedDefaults);
    props.merge(properties);
    return importCircuit(filename, props, callback);
}

std::string PairSynapsesLoader::getName() const
{
    return LOADER_NAME;
}

PropertyMap PairSynapsesLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(PROP_PRESYNAPTIC_NEURON_GID);
    pm.setProperty(PROP_POSTSYNAPTIC_NEURON_GID);
    pm.setProperty(PROP_RADIUS_MULTIPLIER);
    pm.setProperty(PROP_SECTION_TYPE_SOMA);
    pm.setProperty(PROP_SECTION_TYPE_AXON);
    pm.setProperty(PROP_SECTION_TYPE_DENDRITE);
    pm.setProperty(PROP_SECTION_TYPE_APICAL_DENDRITE);
    pm.setProperty(PROP_USE_SDF_SOMA);
    pm.setProperty(PROP_USE_SDF_BRANCHES);
    pm.setProperty(PROP_ASSET_COLOR_SCHEME);
    pm.setProperty(PROP_ASSET_QUALITY);
    return pm;
}
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
