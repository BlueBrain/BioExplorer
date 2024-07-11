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

#include <platform/core/common/loader/Loader.h>

namespace core
{
/** A volume loader for mhd volumes.
 */
class MHDVolumeLoader : public Loader
{
public:
    MHDVolumeLoader(Scene& scene);

    std::vector<std::string> getSupportedStorage() const final;
    std::string getName() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;
    ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

    ModelDescriptorPtr importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                         const PropertyMap& properties) const final;
};

/** A volume loader for raw volumes with params for dimensions.
 */
class RawVolumeLoader : public Loader
{
public:
    RawVolumeLoader(Scene& scene);

    std::vector<std::string> getSupportedStorage() const final;
    std::string getName() const final;
    PropertyMap getProperties() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;
    ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

    ModelDescriptorPtr importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                         const PropertyMap& properties) const final;

private:
    ModelDescriptorPtr _loadVolume(const std::string& filename, const LoaderProgress& callback,
                                   const PropertyMap& properties,
                                   const std::function<void(SharedDataVolumePtr)>& mapData) const;
};
} // namespace core
