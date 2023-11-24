/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include "NeuronsLoader.h"
#include "Neurons.h"

#include <science/common/Logs.h>
#include <science/common/Properties.h>

#include <platform/core/common/Properties.h>

#include <filesystem>

using namespace core;

namespace bioexplorer
{
namespace morphology
{
using namespace common;

const std::string LOADER_NAME = LOADER_NEURONS;
const std::string SUPPORTED_PROTOCOL_NEURONS = "neurons://";

NeuronsLoader::NeuronsLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
}

std::string NeuronsLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> NeuronsLoader::getSupportedStorage() const
{
    return {SUPPORTED_PROTOCOL_NEURONS};
}

bool NeuronsLoader::isSupported(const std::string& storage, const std::string& /*extension*/) const
{
    return (storage.find(SUPPORTED_PROTOCOL_NEURONS) == 0);
}

ModelDescriptorPtr NeuronsLoader::importFromBlob(Blob&& /*blob*/, const LoaderProgress& /*callback*/,
                                                 const PropertyMap& /*properties*/) const
{
    PLUGIN_THROW("Loading Neurons from blob is not supported");
}

ModelDescriptorPtr NeuronsLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                    const PropertyMap& properties) const
{
    PropertyMap props = _defaults;
    props.merge(properties);

    details::NeuronsDetails details;
    const auto baseName = std::filesystem::path(storage).filename();
    details.assemblyName = baseName;
    details.populationName = baseName;
    details.sqlNodeFilter = props.getProperty<std::string>(LOADER_PROPERTY_DATABASE_SQL_FILTER.name);
    details.radiusMultiplier = props.getProperty<double>(LOADER_PROPERTY_RADIUS_MULTIPLIER.name);
    details.populationColorScheme = stringToEnum<morphology::PopulationColorScheme>(
        props.getProperty<std::string>(LOADER_PROPERTY_POPULATION_COLOR_SCHEME.name));
    details.morphologyColorScheme = stringToEnum<morphology::MorphologyColorScheme>(
        props.getProperty<std::string>(LOADER_PROPERTY_MORPHOLOGY_COLOR_SCHEME.name));
    details.realismLevel = 0;
    details.realismLevel += (props.getProperty<bool>(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_SOMA.name)
                                 ? static_cast<int64_t>(morphology::MorphologyRealismLevel::soma)
                                 : 0);
    details.realismLevel += (props.getProperty<bool>(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_DENDRITE.name)
                                 ? static_cast<int64_t>(morphology::MorphologyRealismLevel::dendrite)
                                 : 0);
    details.loadSomas = props.getProperty<bool>(LOADER_PROPERTY_MORPHOLOGY_LOAD_SOMA.name);
    details.loadAxon = props.getProperty<bool>(LOADER_PROPERTY_NEURONS_LOAD_AXON.name);
    details.loadBasalDendrites = props.getProperty<bool>(LOADER_PROPERTY_NEURONS_LOAD_BASAL_DENDRITES.name);
    details.loadApicalDendrites = props.getProperty<bool>(LOADER_PROPERTY_NEURONS_LOAD_APICAL_DENDRITES.name);
    details.generateInternals = props.getProperty<bool>(LOADER_PROPERTY_MORPHOLOGY_GENERATE_INTERNALS.name);
    details.generateExternals = props.getProperty<bool>(LOADER_PROPERTY_NEURONS_GENERATE_EXTERNALS.name);
    details.morphologyRepresentation = stringToEnum<morphology::MorphologyRepresentation>(
        props.getProperty<std::string>(LOADER_PROPERTY_MORPHOLOGY_REPRESENTATION.name));
    details.alignToGrid = props.getProperty<double>(LOADER_PROPERTY_ALIGN_TO_GRID.name);
    details.synapsesType = stringToEnum<morphology::MorphologySynapseType>(
        props.getProperty<std::string>(LOADER_PROPERTY_NEURONS_SYNAPSE_TYPE.name));
    const auto position = props.getProperty<std::array<double, 3>>(LOADER_PROPERTY_POSITION.name);
    const Vector3d pos = core::Vector3d(position[0], position[1], position[2]);
    const auto rotation = props.getProperty<std::array<double, 4>>(LOADER_PROPERTY_ROTATION.name);
    const Quaterniond rot = core::Quaterniond(rotation[3], rotation[0], rotation[1], rotation[2]);
    const auto scale = props.getProperty<std::array<double, 3>>(LOADER_PROPERTY_SCALE.name);
    details.scale = {scale[0], scale[1], scale[2]};
    Neurons Neurons(_scene, details, pos, rot, callback);
    return std::move(Neurons.getModelDescriptor());
}

PropertyMap NeuronsLoader::getProperties() const
{
    return _defaults;
}

PropertyMap NeuronsLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(LOADER_PROPERTY_DATABASE_SQL_FILTER);
    pm.setProperty(LOADER_PROPERTY_ALIGN_TO_GRID);
    pm.setProperty(LOADER_PROPERTY_RADIUS_MULTIPLIER);
    pm.setProperty(LOADER_PROPERTY_POPULATION_COLOR_SCHEME);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_COLOR_SCHEME);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REPRESENTATION);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_LOAD_SOMA);
    pm.setProperty(LOADER_PROPERTY_NEURONS_LOAD_AXON);
    pm.setProperty(LOADER_PROPERTY_NEURONS_LOAD_APICAL_DENDRITES);
    pm.setProperty(LOADER_PROPERTY_NEURONS_LOAD_BASAL_DENDRITES);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_GENERATE_INTERNALS);
    pm.setProperty(LOADER_PROPERTY_NEURONS_GENERATE_EXTERNALS);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_SOMA);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_AXON);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_DENDRITE);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_INTERNALS);
    pm.setProperty(LOADER_PROPERTY_NEURONS_REALISM_LEVEL_EXTERNALS);
    pm.setProperty(LOADER_PROPERTY_NEURONS_REALISM_LEVEL_SPINE);
    pm.setProperty(LOADER_PROPERTY_NEURONS_SYNAPSE_TYPE);
    pm.setProperty(LOADER_PROPERTY_POSITION);
    pm.setProperty(LOADER_PROPERTY_ROTATION);
    pm.setProperty(LOADER_PROPERTY_SCALE);
    return pm;
}
} // namespace morphology
} // namespace bioexplorer
