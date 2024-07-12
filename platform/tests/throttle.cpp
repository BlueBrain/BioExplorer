/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <platform/core/common/Timer.h>

#include "../plugins/Rockets/Throttle.h"

struct TestTimer
{
    void start() { timer.start(); }
    void onCalled()
    {
        std::unique_lock<std::mutex> lock(mutex);
        timer.stop();
        called = true;
        condition.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (!called)
            condition.wait(lock);
    }

    void wait_for(const int64_t waitTime)
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait_for(lock, std::chrono::milliseconds(waitTime), [&] { return !called; });
    }

    auto milliseconds() const { return timer.milliseconds(); }
    auto wasCalled() const { return called; }

private:
    bool called = false;
    std::mutex mutex;
    std::condition_variable condition;
    core::Timer timer;
};

TEST_CASE("timeout_not_cleared")
{
    core::Timeout timeout;

    const int64_t waitTime = 5;
    TestTimer timer;
    timer.start();
    timeout.set([&] { timer.onCalled(); }, waitTime);
    timer.wait();

    CHECK_GE(timer.milliseconds(), waitTime);
    CHECK(timer.wasCalled());
}

TEST_CASE("timeout_with_clear")
{
    core::Timeout timeout;

    const int64_t waitTime = 10;
    TestTimer timer;
    timer.start();
    timeout.set([&] { timer.onCalled(); }, waitTime);
    timeout.clear();
    timer.wait_for(waitTime);

    CHECK(!timer.wasCalled());
}

TEST_CASE("timeout_set_while_not_cleared")
{
    core::Timeout timeout;
    std::atomic_bool called{false};
    timeout.set([&called] { called = true; }, 10000);
    CHECK_THROWS_AS(timeout.set([&called] { called = true; }, 10000), std::logic_error);
    timeout.clear();
    CHECK(!called);
}

TEST_CASE("timeout_clear_while_already_done")
{
    core::Timeout timeout;
    TestTimer timer;
    timeout.set([&] { timer.onCalled(); }, 1);
    timer.wait();
    CHECK(timer.wasCalled());
    CHECK_NOTHROW(timeout.clear());
}

TEST_CASE("throttle_spam_limit")
{
    core::Throttle throttle;
    size_t numCalls{0};
    size_t numSpam{0};
    std::atomic_bool done{false};

    const int64_t waitTime = 10;
    while (!done)
    {
        throttle(
            [&]
            {
                ++numCalls;
                if (numCalls == 10)
                    done = true;
            },
            waitTime);
        ++numSpam;
    }

    CHECK_LT(numCalls, numSpam);
}

TEST_CASE("throttle_spam_check_delayed_call")
{
    core::Throttle throttle;
    size_t numCalls{0};
    std::atomic_bool done{false};
    TestTimer timer;

    while (!done)
        throttle(
            [&]
            {
                if (++numCalls > 2)
                {
                    done = true;
                    timer.onCalled();
                }
            },
            5);
    timer.wait();
    CHECK_EQ(numCalls, 3);
}

TEST_CASE("throttle_one")
{
    core::Throttle throttle;
    size_t numCalls{0};

    throttle([&] { ++numCalls; }, 1);
    CHECK_EQ(numCalls, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK_EQ(numCalls, 1);
}
