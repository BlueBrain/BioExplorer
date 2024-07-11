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
#include <platform/core/engineapi/Model.h>

namespace core
{
struct BinaryParam;
}
SERIALIZATION_ACCESS(BinaryParam)

namespace core
{
struct Chunk
{
    std::string id;
};

struct BinaryParam : ModelParams
{
    size_t size{0};   //!< size in bytes of file
    std::string type; //!< file extension or type (MESH, POINTS, CIRCUIT)
    std::string chunksID;
    SERIALIZATION_FRIEND(BinaryParam)
};

/**
 * A task which receives a file blob, triggers loading of the received blob
 * and adds the loaded model to the engines' scene.
 */
class AddModelFromBlobTask : public Task<ModelDescriptorPtr>
{
public:
    AddModelFromBlobTask(const BinaryParam& param, Engine& engine);

    void appendBlob(const std::string& blob);

private:
    void _checkValidity(Engine& engine);
    void _cancel() final { _chunkEvent.set_exception(std::make_exception_ptr(async::task_canceled())); }
    float _progressBytes() const { return CHUNK_PROGRESS_WEIGHT * ((float)_receivedBytes / _param.size); }

    async::event_task<Blob> _chunkEvent;
    async::event_task<ModelDescriptorPtr> _errorEvent;
    std::vector<async::task<ModelDescriptorPtr>> _finishTasks;
    uint8_ts _blob;
    BinaryParam _param;
    size_t _receivedBytes{0};
    const float CHUNK_PROGRESS_WEIGHT{0.5f};
};
} // namespace core
