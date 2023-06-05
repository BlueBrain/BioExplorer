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

#include "MorphologyCollageLoader.h"

#include <common/Logs.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
const std::string LOADER_NAME = "Morphology collage";

MorphologyCollageLoader::MorphologyCollageLoader(Scene &scene, const ApplicationParameters &applicationParameters,
                                                 PropertyMap &&loaderParams)
    : AbstractCircuitLoader(scene, applicationParameters, std::move(loaderParams))
{
    _fixedDefaults.setProperty({PROP_PRESYNAPTIC_NEURON_GID.name, std::string("")});
    _fixedDefaults.setProperty({PROP_POSTSYNAPTIC_NEURON_GID.name, std::string("")});
    _fixedDefaults.setProperty({PROP_GIDS.name, std::string("")});
    _fixedDefaults.setProperty({PROP_REPORT.name, std::string("")});
    _fixedDefaults.setProperty({PROP_REPORT_TYPE.name, enumToString(ReportType::undefined)});
    _fixedDefaults.setProperty({PROP_CIRCUIT_COLOR_SCHEME.name, enumToString(CircuitColorScheme::none)});
    _fixedDefaults.setProperty({PROP_RADIUS_CORRECTION.name, 0.0});
    _fixedDefaults.setProperty({PROP_DAMPEN_BRANCH_THICKNESS_CHANGERATE.name, true});
    _fixedDefaults.setProperty({PROP_USER_DATA_TYPE.name, enumToString(UserDataType::undefined)});
    _fixedDefaults.setProperty({PROP_MORPHOLOGY_MAX_DISTANCE_TO_SOMA.name, std::numeric_limits<double>::max()});
    _fixedDefaults.setProperty({PROP_MESH_FOLDER.name, std::string("")});
    _fixedDefaults.setProperty({PROP_MESH_FILENAME_PATTERN.name, std::string("")});
    _fixedDefaults.setProperty({PROP_MESH_TRANSFORMATION.name, false});
    _fixedDefaults.setProperty({PROP_LOAD_AFFERENT_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_LOAD_EFFERENT_SYNAPSES.name, false});
    _fixedDefaults.setProperty({PROP_INTERNALS.name, false});
    _fixedDefaults.setProperty({PROP_EXTERNALS.name, false});
}

ModelDescriptorPtr MorphologyCollageLoader::importFromFile(const std::string &filename, const LoaderProgress &callback,
                                                           const PropertyMap &properties) const
{
    PLUGIN_INFO("Loading circuit from " << filename);
    callback.updateProgress("Loading circuit ...", 0);
    PropertyMap props = _defaults;
    props.merge(_fixedDefaults);
    props.merge(properties);
    return importCircuit(filename, props, callback);
}

std::string MorphologyCollageLoader::getName() const
{
    return LOADER_NAME;
}

PropertyMap MorphologyCollageLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(PROP_DENSITY);
    pm.setProperty(PROP_TARGETS);
    pm.setProperty(PROP_RADIUS_MULTIPLIER);
    pm.setProperty(PROP_RANDOM_SEED);
    pm.setProperty(PROP_SECTION_TYPE_SOMA);
    pm.setProperty(PROP_SECTION_TYPE_AXON);
    pm.setProperty(PROP_SECTION_TYPE_DENDRITE);
    pm.setProperty(PROP_SECTION_TYPE_APICAL_DENDRITE);
    pm.setProperty(PROP_USE_SDF_SOMA);
    pm.setProperty(PROP_USE_SDF_BRANCHES);
    pm.setProperty(PROP_USE_SDF_NUCLEUS);
    pm.setProperty(PROP_USE_SDF_MITOCHONDRIA);
    pm.setProperty(PROP_USE_SDF_MYELIN_STEATH);
    pm.setProperty(PROP_USE_SDF_SYNAPSES);
    pm.setProperty(PROP_ASSET_COLOR_SCHEME);
    pm.setProperty(PROP_ASSET_QUALITY);
    pm.setProperty(PROP_CELL_CLIPPING);
    pm.setProperty(PROP_AREAS_OF_INTEREST);
    return pm;
}
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
