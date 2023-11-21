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

#include "AtlasLoader.h"
#include "Atlas.h"

#include <science/common/Logs.h>
#include <science/common/Properties.h>

#include <platform/core/common/Properties.h>

#include <boost/filesystem.hpp>

using namespace core;

namespace bioexplorer
{
namespace atlas
{
using namespace common;

const std::string LOADER_NAME = LOADER_ATLAS;
const std::string SUPPORTED_PROTOCOL_ATLAS = "atlas://";

AtlasLoader::AtlasLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
}

std::string AtlasLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> AtlasLoader::getSupportedStorage() const
{
    return {SUPPORTED_PROTOCOL_ATLAS};
}

bool AtlasLoader::isSupported(const std::string& storage, const std::string& /*extension*/) const
{
    return (storage.find(SUPPORTED_PROTOCOL_ATLAS) == 0);
}

ModelDescriptorPtr AtlasLoader::importFromBlob(Blob&& /*blob*/, const LoaderProgress& /*callback*/,
                                               const PropertyMap& /*properties*/) const
{
    PLUGIN_THROW("Loading atlas from blob is not supported");
}

ModelDescriptorPtr AtlasLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                  const PropertyMap& properties) const
{
    PropertyMap props = _defaults;
    props.merge(properties);

    details::AtlasDetails details;
    const auto baseName = boost::filesystem::basename(storage);
    details.assemblyName = baseName;
    details.cellSqlFilter = props.getProperty<std::string>(LOADER_PROPERTY_ATLAS_CELL_SQL_FILTER.name);
    details.regionSqlFilter = props.getProperty<std::string>(LOADER_PROPERTY_ATLAS_REGION_SQL_FILTER.name);
    details.loadCells = props.getProperty<bool>(LOADER_PROPERTY_ATLAS_LOAD_CELLS.name);
    details.loadMeshes = props.getProperty<bool>(LOADER_PROPERTY_ATLAS_LOAD_MESHES.name);
    details.cellRadius = props.getProperty<double>(LOADER_PROPERTY_ATLAS_CELL_RADIUS.name);
    const auto position = properties.getProperty<std::array<double, 3>>(LOADER_PROPERTY_POSITION.name);
    const Vector3d pos = core::Vector3d(position[0], position[1], position[2]);
    const auto rotation = properties.getProperty<std::array<double, 4>>(LOADER_PROPERTY_ROTATION.name);
    const Quaterniond rot = core::Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]);
    const auto scale = properties.getProperty<std::array<double, 3>>(LOADER_PROPERTY_SCALE.name);
    Atlas atlas(_scene, details, pos, rot, callback);
    return std::move(atlas.getModelDescriptor());
}

PropertyMap AtlasLoader::getProperties() const
{
    return _defaults;
}

PropertyMap AtlasLoader::getCLIProperties()
{
    PropertyMap pm(LOADER_NAME);
    pm.setProperty(LOADER_PROPERTY_ATLAS_LOAD_CELLS);
    pm.setProperty(LOADER_PROPERTY_ATLAS_CELL_SQL_FILTER);
    pm.setProperty(LOADER_PROPERTY_ATLAS_CELL_RADIUS);
    pm.setProperty(LOADER_PROPERTY_ATLAS_LOAD_MESHES);
    pm.setProperty(LOADER_PROPERTY_ATLAS_REGION_SQL_FILTER);
    pm.setProperty(LOADER_PROPERTY_POSITION);
    pm.setProperty(LOADER_PROPERTY_ROTATION);
    pm.setProperty(LOADER_PROPERTY_SCALE);
    return pm;
}
} // namespace atlas
} // namespace bioexplorer
