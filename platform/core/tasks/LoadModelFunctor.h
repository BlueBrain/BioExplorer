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

#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>
#include <platform/core/common/tasks/TaskFunctor.h>
#include <platform/core/engineapi/Model.h>

namespace core
{
/**
 * A task functor which loads data from blob or file path and adds the loaded
 * data to the scene.
 */
class LoadModelFunctor : public TaskFunctor
{
public:
    LoadModelFunctor(Engine& engine, const ModelParams& params);
    LoadModelFunctor(LoadModelFunctor&&) = default;
    ModelDescriptorPtr operator()(Blob&& blob);
    ModelDescriptorPtr operator()();

private:
    ModelDescriptorPtr _performLoad(const std::function<ModelDescriptorPtr()>& loadData);

    ModelDescriptorPtr _loadData(Blob&& blob, const ModelParams& params);
    ModelDescriptorPtr _loadData(const std::string& path, const ModelParams& params);

    void _updateProgress(const std::string& message, const size_t increment);

    std::function<void(std::string, float)> _getProgressFunc();

    Engine& _engine;
    ModelParams _params;
    size_t _currentProgress{0};
    size_t _nextTic{0};
};
} // namespace core
