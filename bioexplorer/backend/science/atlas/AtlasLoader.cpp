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

#include "AtlasLoader.h"
#include "Atlas.h"

#include <science/common/Logs.h>
#include <science/common/Properties.h>

#include <platform/core/common/Properties.h>

#include <filesystem>

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
    const auto baseName = std::filesystem::path(storage).filename();
    details.assemblyName = baseName;
    details.cellSqlFilter = props.getProperty<std::string>(LOADER_PROPERTY_ATLAS_CELL_SQL_FILTER.name);
    details.regionSqlFilter = props.getProperty<std::string>(LOADER_PROPERTY_ATLAS_REGION_SQL_FILTER.name);
    details.loadCells = props.getProperty<bool>(LOADER_PROPERTY_ATLAS_LOAD_CELLS.name);
    details.loadMeshes = props.getProperty<bool>(LOADER_PROPERTY_ATLAS_LOAD_MESHES.name);
    details.cellRadius = props.getProperty<double>(LOADER_PROPERTY_ATLAS_CELL_RADIUS.name);
    const auto position = props.getProperty<std::array<double, 3>>(LOADER_PROPERTY_POSITION.name);
    const Vector3d pos = core::Vector3d(position[0], position[1], position[2]);
    const auto rotation = props.getProperty<std::array<double, 4>>(LOADER_PROPERTY_ROTATION.name);
    const Quaterniond rot = core::Quaterniond(rotation[3], rotation[0], rotation[1], rotation[2]);
    const auto scale = props.getProperty<std::array<double, 3>>(LOADER_PROPERTY_SCALE.name);
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
