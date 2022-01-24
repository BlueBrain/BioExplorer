#include <gtest/gtest.h>
#include <uvw.hpp>

TEST(Timer, StartAndStop)
{
    auto loop = uvw::Loop::getDefault();
    auto handleNoRepeat = loop->resource<uvw::TimerHandle>();
    auto handleRepeat = loop->resource<uvw::TimerHandle>();

    bool checkErrorEvent = false;
    bool checkTimerNoRepeatEvent = false;
    bool checkTimerRepeatEvent = false;

    handleNoRepeat->on<uvw::ErrorEvent>(
        [&checkErrorEvent](const auto &, auto &) {
            ASSERT_FALSE(checkErrorEvent);
            checkErrorEvent = true;
        });

    handleRepeat->on<uvw::ErrorEvent>([&checkErrorEvent](const auto &, auto &) {
        ASSERT_FALSE(checkErrorEvent);
        checkErrorEvent = true;
    });

    handleNoRepeat->on<uvw::TimerEvent>(
        [&checkTimerNoRepeatEvent](const auto &, auto &handle) {
            ASSERT_FALSE(checkTimerNoRepeatEvent);
            checkTimerNoRepeatEvent = true;
            handle.stop();
            handle.close();
            ASSERT_TRUE(handle.closing());
        });

    handleRepeat->on<uvw::TimerEvent>(
        [&checkTimerRepeatEvent](const auto &, auto &handle) {
            if (checkTimerRepeatEvent)
            {
                handle.stop();
                handle.close();
                ASSERT_TRUE(handle.closing());
            }
            else
            {
                checkTimerRepeatEvent = true;
                ASSERT_FALSE(handle.closing());
            }
        });

    handleNoRepeat->start(uvw::TimerHandle::Time{0}, uvw::TimerHandle::Time{0});
    handleRepeat->start(uvw::TimerHandle::Time{0}, uvw::TimerHandle::Time{1});

    ASSERT_TRUE(handleNoRepeat->active());
    ASSERT_FALSE(handleNoRepeat->closing());

    ASSERT_TRUE(handleRepeat->active());
    ASSERT_FALSE(handleRepeat->closing());

    loop->run();

    ASSERT_FALSE(checkErrorEvent);
    ASSERT_TRUE(checkTimerNoRepeatEvent);
    ASSERT_TRUE(checkTimerRepeatEvent);
}

TEST(Timer, Again)
{
    auto loop = uvw::Loop::getDefault();
    auto handle = loop->resource<uvw::TimerHandle>();

    bool checkErrorEvent = false;
    bool checkTimerEvent = true;

    handle->on<uvw::ErrorEvent>([&checkErrorEvent](const auto &, auto &) {
        ASSERT_FALSE(checkErrorEvent);
        checkErrorEvent = true;
    });

    handle->on<uvw::TimerEvent>([&checkTimerEvent](const auto &, auto &handle) {
        static bool guard = false;

        if (guard)
        {
            handle.stop();
            handle.close();
            checkTimerEvent = true;
            ASSERT_TRUE(handle.closing());
        }
        else
        {
            guard = true;
            handle.again();
            ASSERT_EQ(handle.repeat(), uvw::TimerHandle::Time{1});
            ASSERT_FALSE(handle.closing());
        }
    });

    ASSERT_NO_THROW(handle->again());
    ASSERT_FALSE(handle->active());
    ASSERT_TRUE(checkErrorEvent);

    checkErrorEvent = false;
    handle->start(uvw::TimerHandle::Time{0}, uvw::TimerHandle::Time{1});

    ASSERT_TRUE(handle->active());
    ASSERT_FALSE(handle->closing());

    loop->run();

    ASSERT_FALSE(checkErrorEvent);
    ASSERT_TRUE(checkTimerEvent);
}

TEST(Timer, Repeat)
{
    auto loop = uvw::Loop::getDefault();
    auto handle = loop->resource<uvw::TimerHandle>();

    ASSERT_NO_THROW(handle->repeat(uvw::TimerHandle::Time{42}));
    ASSERT_EQ(handle->repeat(), uvw::TimerHandle::Time{42});
}

TEST(Timer, Fake)
{
    auto loop = uvw::Loop::getDefault();
    auto handle = loop->resource<uvw::TimerHandle>();

    auto l = [](const auto &, auto &) { ASSERT_FALSE(true); };
    handle->on<uvw::ErrorEvent>(l);
    handle->on<uvw::TimerEvent>(l);

    handle->start(uvw::TimerHandle::Time{0}, uvw::TimerHandle::Time{0});
    handle->close();

    ASSERT_FALSE(handle->active());
    ASSERT_TRUE(handle->closing());

    loop->run();
}
