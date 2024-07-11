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

#pragma once

#include <chrono>

namespace core
{
using clock = std::chrono::high_resolution_clock;

/** Simple timer class to measure time spent in a portion of the code */
class Timer
{
public:
    Timer();

    /** (Re)Start the timer at 'now' */
    void start();

    /** Stops the timer and records the interval + a smoothed value over time*/
    void stop();

    /** @return the elapsed time in seconds since the last start(). */
    double elapsed() const;

    /** @return last interval from start() to stop() in microseconds. */
    int64_t microseconds() const;

    /** @return last interval from start() to stop() in milliseconds. */
    int64_t milliseconds() const;

    /** @return last interval from start() to stop() in seconds. */
    double seconds() const;

    /**
     * @return last interval from start() to stop() in per seconds, e.g. for
     * frame per seconds
     */
    double perSecond() const;

    /**
     * @return the current FPS, updated every 150 ms
     */
    double fps() const;

    /**
     * @return last smoothed interval from start() to stop() in per seconds,
     * e.g. for frame per seconds
     */
    double perSecondSmoothed() const;

private:
    clock::time_point _startTime;
    int64_t _microseconds{0};
    double _smoothNom{0.0};
    double _smoothDen{0.0};
    const double _smoothingFactor{0.9}; // closer to 1 means more smoothing
    clock::time_point _lastFPSTickTime;
    double _fps{0.0};
};
} // namespace core
