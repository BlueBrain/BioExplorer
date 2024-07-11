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

#pragma once

#include <platform/core/common/PropertyMap.h>
#include <platform/core/common/loader/Loader.h>

#include <set>

namespace core
{
struct LoaderInfo
{
    std::string name;
    std::vector<std::string> extensions;
    PropertyMap properties;
};

/**
 * Holds information about registered loaders and helps invoking the appropriate
 * loader for a given blob or file.
 */
class LoaderRegistry
{
public:
    /** Register the given loader. */
    void registerLoader(std::unique_ptr<Loader> loader);

    /**
     * Get a list of loaders and their supported file extensions and properties
     */
    const std::vector<LoaderInfo>& getLoaderInfos() const;

    /**
     * @return true if any of the registered loaders can handle the given file
     */
    bool isSupportedFile(const std::string& filename) const;

    /**
     * @return true if any of the registered loaders can handle the given type
     */
    bool isSupportedType(const std::string& type) const;

    /**
     * Get a loader that matches the provided name, filetype or loader name.
     * @throw std::runtime_error if no loader found.
     */
    const Loader& getSuitableLoader(const std::string& filename, const std::string& filetype,
                                    const std::string& loaderName) const;

    /**
     * Load the given file or folder into the given scene by choosing the first
     * matching loader based on the filename or filetype.
     *
     * @param path the file or folder containing the data to import
     * @param scene the scene where to add the loaded model to
     * @param transformation the transformation to apply for the added model
     * @param materialID the default material ot use
     * @param cb the callback for progress updates from the loader
     */
    void load(const std::string& path, Scene& scene, const Matrix4f& transformation, const size_t materialID,
              LoaderProgress cb);

    /** @internal */
    void clear();

    /** @internal */
    void registerArchiveLoader(std::unique_ptr<Loader> loader);

private:
    bool _archiveSupported(const std::string& filename, const std::string& filetype) const;

    std::vector<std::unique_ptr<Loader>> _loaders;
    std::unique_ptr<Loader> _archiveLoader;
    std::vector<LoaderInfo> _loaderInfos;
};
} // namespace core
