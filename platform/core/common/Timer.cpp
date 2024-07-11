/*
    Copyright 2017 - 2024 Blue Brain Project / EPFL

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

#include "Timer.h"

#include <chrono>
constexpr double MICRO_PER_SEC = 1000000.0;
constexpr double FPS_UPDATE_MILLISECS = 150;

namespace core
{
Timer::Timer()
{
    _lastFPSTickTime = clock::now();
    _startTime = clock::now();
}

void Timer::start()
{
    _startTime = clock::now();
}

double Timer::elapsed() const
{
    return std::chrono::duration<double>{clock::now() - _startTime}.count();
}

void Timer::stop()
{
    const auto now = clock::now();
    _microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - _startTime).count();
    _smoothNom = _smoothNom * _smoothingFactor + _microseconds / MICRO_PER_SEC;
    _smoothDen = _smoothDen * _smoothingFactor + 1.f;

    const auto secsLastFPSTick = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastFPSTickTime).count();

    if (secsLastFPSTick >= FPS_UPDATE_MILLISECS)
    {
        _lastFPSTickTime = now;
        _fps = perSecond();
    }
}

int64_t Timer::microseconds() const
{
    return _microseconds;
}

int64_t Timer::milliseconds() const
{
    return _microseconds / 1000.0;
}

double Timer::seconds() const
{
    return _microseconds / MICRO_PER_SEC;
}

double Timer::perSecond() const
{
    return MICRO_PER_SEC / _microseconds;
}

double Timer::perSecondSmoothed() const
{
    return _smoothDen / _smoothNom;
}

double Timer::fps() const
{
    return _fps;
}
} // namespace core
