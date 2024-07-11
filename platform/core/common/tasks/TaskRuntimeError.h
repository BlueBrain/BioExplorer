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

#include <stdexcept>

namespace core
{
/**
 * An exception type that shall be thrown at any point during the task execution
 * to provide useful errors for the user.
 */
class TaskRuntimeError : public std::runtime_error
{
public:
    TaskRuntimeError(const std::string& message, const int code_ = -1, const std::string& data_ = "")
        : std::runtime_error(message)
        , code(code_)
        , data(data_)
    {
    }

    const int code;
    const std::string data;
};
} // namespace core
