/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/common/loader/Loader.h>
#include <platform/core/common/loader/LoaderRegistry.h>

#include <set>

namespace core
{
class ArchiveLoader : public Loader
{
public:
    ArchiveLoader(Scene& scene, LoaderRegistry& registry);

    std::vector<std::string> getSupportedStorage() const final;
    std::string getName() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;
    ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

    ModelDescriptorPtr importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                         const PropertyMap& properties) const final;

private:
    ModelDescriptorPtr loadExtracted(const std::string& path, const LoaderProgress& callback,
                                     const PropertyMap& properties) const;
    LoaderRegistry& _registry;
};
} // namespace core
