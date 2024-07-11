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

#pragma once

#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>

namespace bioexplorer
{
namespace connectomics
{
/**
 * Load WhiteMatter from file, memory or database
 */
class WhiteMatterLoader : public core::Loader
{
public:
    WhiteMatterLoader(core::Scene& scene, core::PropertyMap&& loaderParams = {});

    /** @copydoc Loader::getName */
    std::string getName() const final;

    /** @copydoc Loader::getSupportedStorage */
    strings getSupportedStorage() const final;

    /** @copydoc Loader::isSupported */
    bool isSupported(const std::string& storage, const std::string& extension) const final;

    /** @copydoc Loader::getCLIProperties */
    static core::PropertyMap getCLIProperties();

    /** @copydoc Loader::getProperties */
    core::PropertyMap getProperties() const final;

    /** @copydoc Loader::importFromBlob */
    core::ModelDescriptorPtr importFromBlob(core::Blob&& blob, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final;

    /** @copydoc Loader::importFromStorage */
    core::ModelDescriptorPtr importFromStorage(const std::string& storage, const core::LoaderProgress& callback,
                                               const core::PropertyMap& properties) const final;

private:
    core::PropertyMap _defaults;
};
} // namespace connectomics
} // namespace bioexplorer
