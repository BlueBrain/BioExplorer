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

#include <platform/core/common/Logs.h>
#include <platform/core/tasks/AddModelFromBlobTask.h>
#include <platform/core/tasks/Errors.h>

#include <rockets/jsonrpc/types.h>

namespace core
{
const std::string METHOD_REQUEST_MODEL_UPLOAD = "request-model-upload";

/**
 * Manage requests for the request-model-upload RPC by receiving and delegating
 * the blobs to the correct request.
 */
class BinaryRequests
{
public:
    /**
     * Create and remember the AddModelFromBlobTask for upcoming receives of
     * binary data to delegate them to the task.
     */
    auto createTask(const BinaryParam& param, uintptr_t clientID, Engine& engine)
    {
        auto task = std::make_shared<AddModelFromBlobTask>(param, engine);

        std::lock_guard<std::mutex> lock(_mutex);
        _requests.emplace(std::make_pair(clientID, param.chunksID), task);
        _nextChunkID = param.chunksID;

        return task;
    }

    void setNextChunkID(const std::string& id) { _nextChunkID = id; }
    /** The receive and delegate of blobs to the AddModelFromBlobTask. */
    rockets::ws::Response processMessage(const rockets::ws::Request& wsRequest)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto key = std::make_pair(wsRequest.clientID, _nextChunkID);
        if (_requests.count(key) == 0)
        {
            CORE_ERROR("Missing RPC " << METHOD_REQUEST_MODEL_UPLOAD << " or cancelled?");
            return {};
        }

        _requests[key]->appendBlob(wsRequest.message);
        return {};
    }

    /** Remove pending request in case the client connection closed. */
    void removeRequest(const uintptr_t clientID)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto i = _requests.begin(); i != _requests.end();)
        {
            if (i->first.first != clientID)
            {
                ++i;
                continue;
            }
            i->second->cancel();
            i = _requests.erase(i);
        }
    }

    /** Remove finished task. */
    void removeTask(TaskPtr task)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto i = _requests.begin(); i != _requests.end();)
        {
            if (i->second != task)
            {
                ++i;
                continue;
            }
            i = _requests.erase(i);
        }
    }

private:
    using ClientRequestID = std::pair<uintptr_t, std::string>;
    std::map<ClientRequestID, std::shared_ptr<AddModelFromBlobTask>> _requests;
    std::string _nextChunkID;
    std::mutex _mutex;
};
} // namespace core
