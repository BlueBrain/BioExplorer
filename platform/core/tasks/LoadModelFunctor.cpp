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

#include "LoadModelFunctor.h"

#include "Errors.h"

#include <platform/core/common/utils/Utils.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

namespace core
{
const float TOTAL_PROGRESS = 100.f;

LoadModelFunctor::LoadModelFunctor(Engine& engine, const ModelParams& params)
    : _engine(engine)
    , _params(params)
{
}

ModelDescriptorPtr LoadModelFunctor::operator()(Blob&& blob)
{
    return _performLoad([&] { return _loadData(std::move(blob), _params); });
}

ModelDescriptorPtr LoadModelFunctor::operator()()
{
    const auto& path = _params.getPath();
    return _performLoad([&] { return _loadData(path, _params); });
}

ModelDescriptorPtr LoadModelFunctor::_performLoad(const std::function<ModelDescriptorPtr()>& loadData)
{
    try
    {
        return loadData();
    }
    catch (const std::exception& e)
    {
        progress("Loading failed", (TOTAL_PROGRESS - _currentProgress) / TOTAL_PROGRESS, 1.f);
        throw LOADING_BINARY_FAILED(e.what());
    }
}

ModelDescriptorPtr LoadModelFunctor::_loadData(Blob&& blob, const ModelParams& params)
{
    return _engine.getScene().loadModel(std::move(blob), params, {_getProgressFunc()});
}

ModelDescriptorPtr LoadModelFunctor::_loadData(const std::string& path, const ModelParams& params)
{
    return _engine.getScene().loadModel(path, params, {_getProgressFunc()});
}

void LoadModelFunctor::_updateProgress(const std::string& message, const size_t increment)
{
    _currentProgress += increment;
    progress(message, increment / TOTAL_PROGRESS, _currentProgress / TOTAL_PROGRESS);
}

std::function<void(std::string, float)> LoadModelFunctor::_getProgressFunc()
{
    return [this](const std::string& msg, const float progress)
    {
        cancelCheck();
        const size_t newProgress = progress * TOTAL_PROGRESS;
        if (newProgress == 0 || newProgress % size_t(TOTAL_PROGRESS) > _nextTic)
        {
            _updateProgress(msg, newProgress - _nextTic);
            _nextTic = newProgress;
        }
    };
}
} // namespace core
