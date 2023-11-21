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

#include "WhiteMatterLoader.h"
#include "WhiteMatter.h"

#include <science/common/Logs.h>
#include <science/common/Properties.h>

#include <platform/core/common/Properties.h>

#include <boost/filesystem.hpp>

using namespace core;

namespace bioexplorer
{
namespace connectomics
{
using namespace common;

const std::string LOADER_NAME = LOADER_WHITE_MATTER;
const std::string SUPPORTED_PROTOCOL_WHITE_MATTER = "whitematter://";

WhiteMatterLoader::WhiteMatterLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
}

std::string WhiteMatterLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> WhiteMatterLoader::getSupportedStorage() const
{
    return {SUPPORTED_PROTOCOL_WHITE_MATTER};
}

bool WhiteMatterLoader::isSupported(const std::string& storage, const std::string& /*extension*/) const
{
    return (storage.find(SUPPORTED_PROTOCOL_WHITE_MATTER) == 0);
}

ModelDescriptorPtr WhiteMatterLoader::importFromBlob(Blob&& /*blob*/, const LoaderProgress& /*callback*/,
                                                     const PropertyMap& /*properties*/) const
{
    PLUGIN_THROW("Loading atlas from blob is not supported");
}

ModelDescriptorPtr WhiteMatterLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                        const PropertyMap& properties) const
{
    PropertyMap props = _defaults;
    props.merge(properties);

    details::WhiteMatterDetails details;
    const auto baseName = boost::filesystem::basename(storage);
    details.assemblyName = baseName;
    details.populationName = baseName;
    details.sqlFilter = props.getProperty<std::string>(LOADER_PROPERTY_DATABASE_SQL_FILTER.name);
    details.radius = props.getProperty<double>(LOADER_PROPERTY_RADIUS_MULTIPLIER.name);
    const auto position = props.getProperty<std::array<double, 3>>(LOADER_PROPERTY_POSITION.name);
    const Vector3d pos = core::Vector3d(position[0], position[1], position[2]);
    const auto rotation = props.getProperty<std::array<double, 4>>(LOADER_PROPERTY_ROTATION.name);
    const Quaterniond rot = core::Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]);
    const auto scale = props.getProperty<std::array<double, 3>>(LOADER_PROPERTY_SCALE.name);
    WhiteMatter whiteMatter(_scene, details, pos, rot, callback);
    return std::move(whiteMatter.getModelDescriptor());
}

PropertyMap WhiteMatterLoader::getProperties() const
{
    return _defaults;
}

PropertyMap WhiteMatterLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(LOADER_PROPERTY_DATABASE_SQL_FILTER);
    pm.setProperty(LOADER_PROPERTY_RADIUS_MULTIPLIER);
    pm.setProperty(LOADER_PROPERTY_POSITION);
    pm.setProperty(LOADER_PROPERTY_ROTATION);
    pm.setProperty(LOADER_PROPERTY_SCALE);
    return pm;
}
} // namespace connectomics
} // namespace bioexplorer
