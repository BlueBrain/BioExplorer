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
#include <platform/core/common/tasks/Task.h>

namespace core
{
/**
 * A task which loads data from the path of the given params and adds the loaded
 * model to the engines' scene.
 */
class AddModelTask : public Task<ModelDescriptorPtr>
{
public:
    AddModelTask(const ModelParams& model, Engine& engine);
};
} // namespace core
