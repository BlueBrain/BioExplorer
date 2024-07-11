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

#include <platform/core/common/tasks/Task.h>
#include <platform/core/common/tasks/TaskRuntimeError.h>

namespace core
{
const auto ERROR_ID_MISSING_PARAMS = -1731;
const auto ERROR_ID_UNSUPPORTED_TYPE = -1732;
const auto ERROR_ID_INVALID_BINARY_RECEIVE = -1733;
const auto ERROR_ID_LOADING_BINARY_FAILED = -1734;

const TaskRuntimeError MISSING_PARAMS{"Missing params", ERROR_ID_MISSING_PARAMS};

const TaskRuntimeError UNSUPPORTED_TYPE{"Unsupported type", ERROR_ID_UNSUPPORTED_TYPE};

const TaskRuntimeError INVALID_BINARY_RECEIVE{
    "Invalid binary received; no more files expected or "
    "current file is complete",
    ERROR_ID_INVALID_BINARY_RECEIVE};

inline TaskRuntimeError LOADING_BINARY_FAILED(const std::string& error)
{
    return {error, ERROR_ID_LOADING_BINARY_FAILED};
}
} // namespace core
