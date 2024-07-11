/*
    Copyright 2015 - 2018 Blue Brain Project / EPFL

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

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>

namespace core
{
/**
 * Implements the setTimeout() and clearTimeout() mechanics from Javascript
 * (https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/setTimeout),
 * following the same semantics.
 */
struct Timeout
{
    ~Timeout();
    void set(const std::function<void()>& func, const int64_t wait);
    void clear();

private:
    std::mutex _mutex;
    std::condition_variable _condition;
    std::future<void> _timeout;
    std::atomic_bool _cleared{true};
};
} // namespace core
