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

#include "Throttle.h"

namespace core
{
namespace
{
using namespace std::chrono;
auto now()
{
    return high_resolution_clock::now();
}
auto elapsedSince(const time_point<high_resolution_clock>& last)
{
    return duration_cast<milliseconds>(now() - last).count();
}
} // namespace

void Throttle::operator()(const Throttle::Function& fn, const int64_t wait)
{
    operator()(fn, fn, wait);
}

void Throttle::operator()(const Throttle::Function& fn, const Throttle::Function& later, const int64_t wait)
{
    time_point last;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        last = _last;
    }
    if (_haveLast && (elapsedSince(last) <= wait))
    {
        _timeout.clear();
        auto delayed = [&_last = _last, &mutex = _mutex, later]
        {
            std::lock_guard<std::mutex> lock(mutex);
            later();
            _last = now();
        };
        _timeout.set(delayed, wait);
    }
    else
    {
        fn();
        _haveLast = true;
        std::lock_guard<std::mutex> lock(_mutex);
        _last = now();
    }
}
} // namespace core
