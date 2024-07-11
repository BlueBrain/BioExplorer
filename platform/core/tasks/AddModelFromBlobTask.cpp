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

#include "AddModelFromBlobTask.h"

#include "Errors.h"
#include "LoadModelFunctor.h"

#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>

#include <sstream>

namespace core
{
AddModelFromBlobTask::AddModelFromBlobTask(const BinaryParam& param, Engine& engine)
    : _param(param)
{
    _checkValidity(engine);

    _blob.reserve(param.size);

    LoadModelFunctor functor{engine, param};
    functor.setCancelToken(_cancelToken);
    functor.setProgressFunc([&progress = progress, w = CHUNK_PROGRESS_WEIGHT](const auto& msg, auto, auto amount)
                            { progress.update(msg, w + (amount * (1.f - w))); });

    // load data, return model descriptor or stop if blob receive was invalid
    _finishTasks.emplace_back(_errorEvent.get_task());
    _finishTasks.emplace_back(_chunkEvent.get_task().then(std::move(functor)));
    _task = async::when_any(_finishTasks)
                .then(
                    [&engine](async::when_any_result<std::vector<async::task<ModelDescriptorPtr>>> results)
                    {
                        engine.triggerRender();
                        return results.tasks[results.index].get();
                    });
}

void AddModelFromBlobTask::appendBlob(const std::string& blob)
{
    // if more bytes than expected are received, error and stop
    if (_blob.size() + blob.size() > _param.size)
    {
        _errorEvent.set_exception(std::make_exception_ptr(INVALID_BINARY_RECEIVE));
        return;
    }

    _blob.insert(_blob.end(), blob.begin(), blob.end());

    _receivedBytes += blob.size();
    std::stringstream msg;
    msg << "Receiving " << _param.getName() << " ...";
    progress.update(msg.str(), _progressBytes());

    // if blob is complete, start the loading
    if (_blob.size() == _param.size)
        _chunkEvent.set({_param.type, _param.getName(), std::move(_blob)});
}

void AddModelFromBlobTask::_checkValidity(Engine& engine)
{
    if (_param.type.empty() || _param.size == 0)
        throw MISSING_PARAMS;

    const auto& registry = engine.getScene().getLoaderRegistry();
    if (!registry.isSupportedType(_param.type))
        throw UNSUPPORTED_TYPE;
}
} // namespace core
