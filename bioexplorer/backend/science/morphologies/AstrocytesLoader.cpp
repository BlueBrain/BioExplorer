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

#include "AstrocytesLoader.h"
#include "Astrocytes.h"

#include <science/common/Logs.h>
#include <science/common/Properties.h>

#include <platform/core/common/Properties.h>

#include <boost/filesystem.hpp>

using namespace core;

namespace bioexplorer
{
namespace morphology
{
using namespace common;

const std::string LOADER_NAME = LOADER_ASTROCYTES;
const std::string SUPPORTED_PROTOCOL_ASTROCYTES = "astrocytes://";

AstrocytesLoader::AstrocytesLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
}

std::string AstrocytesLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> AstrocytesLoader::getSupportedStorage() const
{
    return {SUPPORTED_PROTOCOL_ASTROCYTES};
}

bool AstrocytesLoader::isSupported(const std::string& storage, const std::string& /*extension*/) const
{
    return (storage.find(SUPPORTED_PROTOCOL_ASTROCYTES) == 0);
}

ModelDescriptorPtr AstrocytesLoader::importFromBlob(Blob&& /*blob*/, const LoaderProgress& /*callback*/,
                                                    const PropertyMap& /*properties*/) const
{
    PLUGIN_THROW("Loading astrocytes from blob is not supported");
}

ModelDescriptorPtr AstrocytesLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                       const PropertyMap& properties) const
{
    PropertyMap props = _defaults;
    props.merge(properties);

    details::AstrocytesDetails details;
    const auto baseName = boost::filesystem::basename(storage);
    details.assemblyName = baseName;
    details.populationName = baseName;
    details.vasculaturePopulationName =
        props.getProperty<std::string>(LOADER_PROPERTY_ASTROCYTES_VASCULATURE_SCHEMA.name);
    details.sqlFilter = props.getProperty<std::string>(LOADER_PROPERTY_DATABASE_SQL_FILTER.name);
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
    details.loadDendrites = props.getProperty<bool>(LOADER_PROPERTY_MORPHOLOGY_LOAD_DENDRITES.name);
    details.generateInternals = props.getProperty<bool>(LOADER_PROPERTY_MORPHOLOGY_GENERATE_INTERNALS.name);
    details.loadMicroDomains = props.getProperty<bool>(LOADER_PROPERTY_ASTROCYTES_LOAD_MICRO_DOMAINS.name);
    details.morphologyRepresentation = stringToEnum<morphology::MorphologyRepresentation>(
        props.getProperty<std::string>(LOADER_PROPERTY_MORPHOLOGY_REPRESENTATION.name));
    details.alignToGrid = props.getProperty<double>(LOADER_PROPERTY_ALIGN_TO_GRID.name);
    const auto position = properties.getProperty<std::array<double, 3>>(LOADER_PROPERTY_POSITION.name);
    const Vector3d pos = core::Vector3d(position[0], position[1], position[2]);
    const auto rotation = properties.getProperty<std::array<double, 4>>(LOADER_PROPERTY_ROTATION.name);
    const Quaterniond rot = core::Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]);
    const auto scale = properties.getProperty<std::array<double, 3>>(LOADER_PROPERTY_SCALE.name);
    details.scale = {scale[0], scale[1], scale[2]};
    Astrocytes astrocytes(_scene, details, pos, rot, callback);
    return std::move(astrocytes.getModelDescriptor());
}

PropertyMap AstrocytesLoader::getProperties() const
{
    return _defaults;
}

PropertyMap AstrocytesLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(LOADER_PROPERTY_DATABASE_SQL_FILTER);
    pm.setProperty(LOADER_PROPERTY_ALIGN_TO_GRID);
    pm.setProperty(LOADER_PROPERTY_RADIUS_MULTIPLIER);
    pm.setProperty(LOADER_PROPERTY_POPULATION_COLOR_SCHEME);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_COLOR_SCHEME);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REPRESENTATION);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_LOAD_SOMA);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_LOAD_DENDRITES);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_GENERATE_INTERNALS);
    pm.setProperty(LOADER_PROPERTY_ASTROCYTES_LOAD_MICRO_DOMAINS);
    pm.setProperty(LOADER_PROPERTY_ASTROCYTES_VASCULATURE_SCHEMA);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_SOMA);
    pm.setProperty(LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_DENDRITE);
    pm.setProperty(LOADER_PROPERTY_POSITION);
    pm.setProperty(LOADER_PROPERTY_ROTATION);
    pm.setProperty(LOADER_PROPERTY_SCALE);
    return pm;
}
} // namespace morphology
} // namespace bioexplorer
