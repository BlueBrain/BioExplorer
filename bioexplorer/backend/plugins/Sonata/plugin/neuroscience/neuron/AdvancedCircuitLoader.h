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

#include "AbstractCircuitLoader.h"

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
class AdvancedCircuitLoader : public AbstractCircuitLoader
{
public:
    AdvancedCircuitLoader(core::Scene &scene, const core::ApplicationParameters &applicationParameters,
                          core::PropertyMap &&loaderParams);

    std::string getName() const final;

    static core::PropertyMap getCLIProperties();

    core::ModelDescriptorPtr importFromStorage(const std::string &path, const core::LoaderProgress &callback,
                                               const core::PropertyMap &properties) const final;
};
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
