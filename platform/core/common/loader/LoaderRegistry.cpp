/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "LoaderRegistry.h"

#include <platform/core/common/utils/FileSystem.h>
#include <platform/core/common/utils/Utils.h>

namespace core
{
void LoaderRegistry::registerLoader(std::unique_ptr<Loader> loader)
{
    _loaderInfos.push_back({loader->getName(), loader->getSupportedStorage(), loader->getProperties()});
    _loaders.push_back(std::move(loader));
}

const std::vector<LoaderInfo>& LoaderRegistry::getLoaderInfos() const
{
    return _loaderInfos;
}

bool LoaderRegistry::isSupportedFile(const std::string& filename) const
{
    if (fs::is_directory(filename))
        return false;

    const auto extension = extractExtension(filename);
    if (_archiveSupported(filename, extension))
        return true;
    for (const auto& loader : _loaders)
        if (loader->isSupported(filename, extension))
            return true;
    return false;
}

bool LoaderRegistry::isSupportedType(const std::string& type) const
{
    if (_archiveSupported("", type))
        return true;
    for (const auto& loader : _loaders)
        if (loader->isSupported("", type))
            return true;
    return false;
}

const Loader& LoaderRegistry::getSuitableLoader(const std::string& filename, const std::string& filetype,
                                                const std::string& loaderName) const
{
    if (fs::is_directory(filename))
        throw std::runtime_error("'" + filename + "' is a directory");

    const auto extension = filetype.empty() ? extractExtension(filename) : filetype;

    // If we have an archive we always use the archive loader even if a specific
    // loader is specified
    if (_archiveSupported(filename, extension))
        return *_archiveLoader;

    // Find specific loader
    if (!loaderName.empty())
    {
        for (const auto& loader : _loaders)
            if (loader->getName() == loaderName)
                return *loader.get();

        throw std::runtime_error("No loader found with name '" + loaderName + "'");
    }

    for (const auto& loader : _loaders)
        if (loader->isSupported(filename, extension))
            return *loader;

    throw std::runtime_error("No loader found for filename '" + filename + "' and filetype '" + filetype + "'");
}

void LoaderRegistry::clear()
{
    _loaders.clear();
    _archiveLoader.reset();
    _loaderInfos.clear();
}

void LoaderRegistry::registerArchiveLoader(std::unique_ptr<Loader> loader)
{
    _archiveLoader = std::move(loader);
}

bool LoaderRegistry::_archiveSupported(const std::string& filename, const std::string& filetype) const
{
    return _archiveLoader && _archiveLoader->isSupported(filename, filetype);
}
} // namespace core
