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

#include <platform/core/common/BaseObject.h>

#include <functional>
#include <mutex>
#include <string>

namespace core
{
/**
 * A progress object which offers thread-safe progress updates and thread-safe
 * consumption of the current progress if it has changed in between.
 */
class Progress : public BaseObject
{
public:
    Progress() = default;
    explicit Progress(const std::string& operation)
        : _operation(operation)
    {
    }

    /** Update the progress with a new absolute amount. */
    void update(const std::string& operation, const float amount)
    {
        std::lock_guard<std::mutex> lock_(_mutex);
        _updateValue(_operation, operation);
        _updateValue(_amount, amount);
    }

    /** Update the progress with the given increment. */
    void increment(const std::string& operation, const float increment)
    {
        std::lock_guard<std::mutex> lock_(_mutex);
        _updateValue(_operation, operation);
        _updateValue(_amount, _amount + increment);
    }

    /**
     * Call the provided callback with the current progress if it has changed
     * since the last invokation.
     */
    void consume(std::function<void(std::string, float)> callback)
    {
        std::lock_guard<std::mutex> lock_(_mutex);
        if (isModified())
        {
            callback(_operation, _amount);
            resetModified();
        }
    }

private:
    std::string _operation;
    float _amount{0.f};
    std::mutex _mutex;
};
} // namespace core
