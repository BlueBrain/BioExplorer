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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <async++.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <string>

namespace core
{
/**
 * A base class for functors that can be used for Tasks. It provides support for
 * progress reporting and cancellation during execution.
 */
class TaskFunctor
{
public:
    /** message, increment, amount */
    using ProgressFunc = std::function<void(std::string, float, float)>;

    /** Set the function when progress() is called. */
    void setProgressFunc(const ProgressFunc& progressFunc) { _progressFunc = progressFunc; }

    /**
     * Report progress using the provided callback from setProgressFunc() and
     * also check if the execution has been cancelled.
     *
     * @param message the progress message
     * @param increment the fractional increment of this progress update
     * @param amount the absolute amount of progress at the time of this update
     */
    void progress(const std::string& message, const float increment, const float amount)
    {
        cancelCheck();
        if (_progressFunc)
            _progressFunc(message, increment, amount);
    }

    /** Set the cancel token from e.g. the task that uses this functor. */
    void setCancelToken(async::cancellation_token& cancelToken) { _cancelToken = &cancelToken; }

    /**
     * Checks if the execution has been cancelled. If so, this will throw an
     * exception that is ultimately handled by the task and is stored in the
     * tasks' result.
     */
    void cancelCheck() const
    {
        if (_cancelToken)
            async::interruption_point(*_cancelToken);
    }

private:
    async::cancellation_token* _cancelToken{nullptr};
    ProgressFunc _progressFunc;
};
} // namespace core
