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

#include <platform/core/common/Progress.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/tasks/TaskFunctor.h>

namespace core
{
/**
 * A task is an operation that can be scheduled (directly, async, ...) and has
 * support for progress reporting during the execution and cancellation of the
 * execution.
 */
class AbstractTask
{
public:
    virtual ~AbstractTask() = default;

    /**
     * Cancels the task if is either waiting to be scheduled or already running.
     * Will have no effect if the task already finished.
     *
     * @param done for asynchronous cancel processing, this function will be
     *             remembered and can be called via finishCancel()
     */
    void cancel(std::function<void()> done = {})
    {
        if (_cancelled)
            return;
        _cancelDone = done;
        _cancelled = true;
        _cancelToken.cancel();
        _cancel();
    }

    /**
     * Indicate that cancel processing has finished and call the function given
     * to cancel().
     */
    void finishCancel()
    {
        if (_cancelDone)
            _cancelDone();
    }

    /** @return true if the task has been cancelled. */
    bool canceled() const { return _cancelled; }
    /**
     * Schedule the execution of the task. Depending on the concrete task type,
     * the task could still be running though after construction.
     */
    virtual void schedule() = 0;

    /** @return access to the progress of task. */
    Progress progress{"Scheduling task ..."};

protected:
    async::cancellation_token _cancelToken;
    std::function<void()> _cancelDone;
    std::atomic_bool _cancelled{false};

private:
    virtual void _cancel() {}
};

/**
 * A task type which is directly scheduled after creation. Its result after
 * successful execution is of type T.
 *
 * If the functor is of type TaskFunctor, it will be provided with cancel
 * support and progress feedback possibility.
 */
template <typename T>
class Task : public AbstractTask
{
public:
    using Type = async::task<T>;

    /**
     * Create an empty task; use task() and async++ to do something meaningful.
     */
    Task() = default;

    /** Create and schedule a task with the given functor or lambda. */
    template <typename F>
    Task(F&& functor)
    {
        _task = async::spawn(_setupFunctor(std::move(functor)));
    }

    /** NOP for this task; tasks are running after construction. */
    void schedule() override
    { /* task is already running after construction */
    }

    /**
     * @return the result of tasks, or an exception in case of errors or
     *         cancellation.
     */
    T result() { return _task.get(); }
    /** @return access to the async++ task for chaining, assignment, etc. */
    auto& get() { return _task; }

protected:
    Type _task;

    template <typename F>
    auto&& _setupFunctor(F&& functor)
    {
        if (std::is_base_of<TaskFunctor, F>::value)
        {
            auto& taskFunctor = static_cast<TaskFunctor&>(functor);
            taskFunctor.setProgressFunc(
                std::bind(&Progress::update, std::ref(progress), std::placeholders::_1, std::placeholders::_3));
            taskFunctor.setCancelToken(_cancelToken);
        }
        return std::move(functor);
    }
};

/**
 * A task type which allows for deferred scheduling after construction using
 * schedule().
 */
template <typename T>
class DeferredTask : public Task<T>
{
public:
    template <typename F>
    DeferredTask(F&& functor)
    {
        Task<T>::_task = _e.get_task().then(Task<T>::template _setupFunctor(std::move(functor)));
    }

    void schedule() final { _e.set(); }

private:
    async::event_task<void> _e;
};
} // namespace core
