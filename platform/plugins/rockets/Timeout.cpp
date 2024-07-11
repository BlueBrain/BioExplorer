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

#include "Timeout.h"

namespace core
{
Timeout::~Timeout()
{
    clear();
}

void Timeout::set(const std::function<void()>& func, const int64_t wait)
{
    if (_timeout.valid())
        throw std::logic_error("Timeout cannot be set() while it is still active");

    _cleared = false;
    _timeout =
        std::async(std::launch::async,
                   [&mutex = _mutex, &condition = _condition, &cleared = _cleared, wait, func]
                   {
                       std::unique_lock<std::mutex> lock(mutex);
                       while (!cleared) // deals with spurious wakeups
                       {
                           if (condition.wait_for(lock, std::chrono::milliseconds(wait)) == std::cv_status::timeout)
                           {
                               func();
                               break;
                           }
                       }
                   });
};

void Timeout::clear()
{
    _cleared = true;
    if (_timeout.valid())
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _condition.notify_one();
        }
        _timeout.get();
    }
}
} // namespace core
