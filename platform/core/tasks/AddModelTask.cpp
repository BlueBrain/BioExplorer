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

#include "AddModelTask.h"

#include "Errors.h"
#include "LoadModelFunctor.h"

#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

namespace core
{
AddModelTask::AddModelTask(const ModelParams& modelParams, Engine& engine)
{
    const auto& registry = engine.getScene().getLoaderRegistry();

    // pre-check for validity of given paths
    const auto& path = modelParams.getPath();
    if (path.empty())
        throw MISSING_PARAMS;

    if (!registry.isSupportedFile(path))
        throw UNSUPPORTED_TYPE;

    LoadModelFunctor functor{engine, modelParams};
    functor.setCancelToken(_cancelToken);
    functor.setProgressFunc([&progress = progress](const auto& msg, auto, auto amount)
                            { progress.update(msg, amount); });

    // load data, return model descriptor
    _task = async::spawn(std::move(functor))
                .then(
                    [&engine](async::task<ModelDescriptorPtr> result)
                    {
                        engine.triggerRender();
                        return result.get();
                    });
}
} // namespace core
