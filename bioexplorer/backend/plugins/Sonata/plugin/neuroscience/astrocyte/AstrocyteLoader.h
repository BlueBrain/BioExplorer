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

#include <platform/core/common/Transformation.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace astrocyte
{
class AstrocyteLoader : public core::Loader
{
public:
    AstrocyteLoader(core::Scene &scene, const core::ApplicationParameters &applicationParameters,
                    core::PropertyMap &&loaderParams);

    std::string getName() const final;

    strings getSupportedStorage() const final;

    bool isSupported(const std::string &filename, const std::string &extension) const final;

    static core::PropertyMap getCLIProperties();

    /** @copydoc Loader::importFromBlob */
    core::ModelDescriptorPtr importFromBlob(core::Blob &&blob, const core::LoaderProgress &callback,
                                            const core::PropertyMap &properties) const final;

    /** @copydoc Loader::importFromFile */
    core::ModelDescriptorPtr importFromStorage(const std::string &path, const core::LoaderProgress &callback,
                                               const core::PropertyMap &properties) const final;

private:
    void _importMorphologiesFromURIs(const core::PropertyMap &properties, const std::vector<std::string> &uris,
                                     const core::LoaderProgress &callback, core::Model &model) const;
    const core::ApplicationParameters &_applicationParameters;
    core::PropertyMap _defaults;
    core::PropertyMap _fixedDefaults;
};
} // namespace astrocyte
} // namespace neuroscience
} // namespace sonataexplorer
