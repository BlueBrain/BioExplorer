#pragma once

#include "emitter.hpp"
#include <chrono>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <uv.h>

namespace uvw
{
namespace details
{
enum class UVLoopOption : std::underlying_type_t<uv_loop_option>
{
    BLOCK_SIGNAL = UV_LOOP_BLOCK_SIGNAL
};

enum class UVRunMode : std::underlying_type_t<uv_run_mode>
{
    DEFAULT = UV_RUN_DEFAULT,
    ONCE = UV_RUN_ONCE,
    NOWAIT = UV_RUN_NOWAIT
};

} // namespace details

/**
 * @brief Untyped handle class
 *
 * Handles' types are unknown from the point of view of the loop.<br/>
 * Anyway, a loop maintains a list of all the associated handles and let the
 * users walk them as untyped instances.<br/>
 * This can help to end all the pending requests by closing the handles.
 */
class BaseHandle
{
public:
    /**
     * @brief Checks if the handle is active.
     *
     * What _active_ means depends on the type of handle:
     *
     * * An AsyncHandle handle is always active and cannot be deactivated,
     * except by closing it with uv_close().
     * * A PipeHandle, TcpHandle, UDPHandle, etc. handle - basically any handle
     * that deals with I/O - is active when it is doing something that involves
     * I/O, like reading, writing, connecting, accepting new connections, etc.
     * * A CheckHandle, IdleHandle, TimerHandle, etc. handle is active when it
     * has been started with a call to `start()`.
     *
     * Rule of thumb: if a handle of type `FooHandle` has a `start()` member
     * method, then it’s active from the moment that method is called. Likewise,
     * `stop()` deactivates the handle again.
     *
     * @return True if the handle is active, false otherwise.
     */
    virtual bool active() const noexcept = 0;

    /**
     * @brief Checks if a handle is closing or closed.
     *
     * This function should only be used between the initialization of the
     * handle and the arrival of the close callback.
     *
     * @return True if the handle is closing or closed, false otherwise.
     */
    virtual bool closing() const noexcept = 0;

    /**
     * @brief Reference the given handle.
     *
     * References are idempotent, that is, if a handle is already referenced
     * calling this function again will have no effect.
     */
    virtual void reference() noexcept = 0;

    /**
     * @brief Unreference the given handle.
     *
     * References are idempotent, that is, if a handle is not referenced calling
     * this function again will have no effect.
     */
    virtual void unreference() noexcept = 0;

    /**
     * @brief Checks if the given handle referenced.
     * @return True if the handle referenced, false otherwise.
     */
    virtual bool referenced() const noexcept = 0;

    /**
     * @brief Request handle to be closed.
     *
     * This **must** be called on each handle before memory is released.<br/>
     * In-progress requests are cancelled and this can result in an ErrorEvent
     * emitted.
     */
    virtual void close() noexcept = 0;
};

/**
 * @brief The Loop class.
 *
 * The event loop is the central part of `uvw`'s functionalities, as well as
 * libuv's ones.<br/>
 * It takes care of polling for I/O and scheduling callbacks to be run based on
 * different sources of events.
 */
class Loop final : public Emitter<Loop>,
                   public std::enable_shared_from_this<Loop>
{
    using Deleter = void (*)(uv_loop_t *);

    template <typename, typename>
    friend class Resource;

    Loop(std::unique_ptr<uv_loop_t, Deleter> ptr) noexcept
        : loop{std::move(ptr)}
    {
    }

public:
    using Time = std::chrono::milliseconds;
    using Configure = details::UVLoopOption;
    using Mode = details::UVRunMode;

    /**
     * @brief Initializes a new Loop instance.
     * @return A pointer to the newly created loop.
     */
    static std::shared_ptr<Loop> create()
    {
        auto ptr =
            std::unique_ptr<uv_loop_t, Deleter>{new uv_loop_t,
                                                [](uv_loop_t *l) { delete l; }};
        auto loop = std::shared_ptr<Loop>(new Loop{std::move(ptr)});

        if (uv_loop_init(loop->loop.get()))
        {
            loop = nullptr;
        }

        return loop;
    }

    /**
     * @brief Gets the initialized default loop.
     *
     * It may return an empty pointer in case of failure.<br>
     * This function is just a convenient way for having a global loop
     * throughout an application, the default loop is in no way different than
     * the ones initialized with `create()`.<br>
     * As such, the default loop can be closed with `close()` so the resources
     * associated with it are freed (even if it is not strictly necessary).
     *
     * @return The initialized default loop.
     */
    static std::shared_ptr<Loop> getDefault()
    {
        static std::weak_ptr<Loop> ref;
        std::shared_ptr<Loop> loop;

        if (ref.expired())
        {
            auto def = uv_default_loop();

            if (def)
            {
                auto ptr =
                    std::unique_ptr<uv_loop_t, Deleter>(def,
                                                        [](uv_loop_t *) {});
                loop = std::shared_ptr<Loop>(new Loop{std::move(ptr)});
            }

            ref = loop;
        }
        else
        {
            loop = ref.lock();
        }

        return loop;
    }

    Loop(const Loop &) = delete;
    Loop(Loop &&other) = delete;
    Loop &operator=(const Loop &) = delete;
    Loop &operator=(Loop &&other) = delete;

    ~Loop() noexcept
    {
        if (loop)
        {
            close();
        }
    }

    /**
     * @brief Sets additional loop options.
     *
     * You should normally call this before the first call to uv_run() unless
     * mentioned otherwise.<br/>
     * Supported options:
     *
     * * `Loop::Configure::BLOCK_SIGNAL`: Block a signal when polling for new
     * events. A second argument is required and it is the signal number.
     *
     * An ErrorEvent will be emitted in case of errors.
     *
     * See the official
     * [documentation](http://docs.libuv.org/en/v1.x/loop.html#c.uv_loop_configure)
     * for further details.
     */
    template <typename... Args>
    void configure(Configure flag, Args... args)
    {
        auto err = uv_loop_configure(
            loop.get(), static_cast<std::underlying_type_t<Configure>>(flag),
            std::forward<Args>(args)...);
        if (err)
        {
            publish(ErrorEvent{err});
        }
    }

    /**
     * @brief Creates resources of handles' types.
     *
     * This should be used as a default method to create resources.<br/>
     * The arguments are the ones required for the specific resource.
     *
     * Use it as `loop->resource<uvw::TimerHandle>()`.
     *
     * @return A pointer to the newly created resource.
     */
    template <typename R, typename... Args>
    std::enable_if_t<std::is_base_of<BaseHandle, R>::value, std::shared_ptr<R>>
        resource(Args &&... args)
    {
        auto ptr = R::create(shared_from_this(), std::forward<Args>(args)...);
        ptr = ptr->init() ? ptr : nullptr;
        return ptr;
    }

    /**
     * @brief Creates resources of types other than handles' ones.
     *
     * This should be used as a default method to create resources.<br/>
     * The arguments are the ones required for the specific resource.
     *
     * Use it as `loop->resource<uvw::WorkReq>()`.
     *
     * @return A pointer to the newly created resource.
     */
    template <typename R, typename... Args>
    std::enable_if_t<not std::is_base_of<BaseHandle, R>::value,
                     std::shared_ptr<R>>
        resource(Args &&... args)
    {
        return R::create(shared_from_this(), std::forward<Args>(args)...);
    }

    /**
     * @brief Releases all internal loop resources.
     *
     * Call this function only when the loop has finished executing and all open
     * handles and requests have been closed, or the loop will emit an error.
     *
     * An ErrorEvent will be emitted in case of errors.
     */
    void close()
    {
        auto err = uv_loop_close(loop.get());
        if (err)
        {
            publish(ErrorEvent{err});
        }
    }

    /**
     * @brief Runs the event loop.
     *
     * Available modes are:
     *
     * * `Loop::Mode::DEFAULT`: Runs the event loop until there are no more
     * active and referenced handles or requests.
     * * `Loop::Mode::ONCE`: Poll for i/o once. Note that this function blocks
     * if there are no pending callbacks.
     * * `Loop::Mode::NOWAIT`: Poll for i/o once but don’t block if there are no
     * pending callbacks.
     *
     * See the official
     * [documentation](http://docs.libuv.org/en/v1.x/loop.html#c.uv_run)
     * for further details.
     *
     * @return True when done, false in all other cases.
     */
    template <Mode mode = Mode::DEFAULT>
    bool run() noexcept
    {
        auto utm = static_cast<std::underlying_type_t<Mode>>(mode);
        auto uvrm = static_cast<uv_run_mode>(utm);
        return (uv_run(loop.get(), uvrm) == 0);
    }

    /**
     * @brief Checks if there are active resources.
     * @return True if there are active resources in the loop.
     */
    bool alive() const noexcept { return !(uv_loop_alive(loop.get()) == 0); }

    /**
     * @brief Stops the event loop.
     *
     * It causes `run()` to end as soon as possible.<br/>
     * This will happen not sooner than the next loop iteration.<br/>
     * If this function was called before blocking for I/O, the loop won’t block
     * for I/O on this iteration.
     */
    void stop() noexcept { uv_stop(loop.get()); }

    /**
     * @brief Get backend file descriptor.
     *
     * Only kqueue, epoll and event ports are supported.<br/>
     * This can be used in conjunction with `run<Loop::Mode::NOWAIT>()` to poll
     * in one thread and run the event loop’s callbacks in another.
     *
     * @return The backend file descriptor.
     */
    int descriptor() const noexcept { return uv_backend_fd(loop.get()); }

    /**
     * @brief Gets the poll timeout.
     * @return The return value is in milliseconds, or -1 for no timeout.
     */
    Time timeout() const noexcept
    {
        return Time{uv_backend_timeout(loop.get())};
    }

    /**
     * @brief Returns the current timestamp in milliseconds.
     *
     * The timestamp is cached at the start of the event loop tick.<br/>
     * The timestamp increases monotonically from some arbitrary point in
     * time.<br/>
     * Don’t make assumptions about the starting point, you will only get
     * disappointed.
     *
     * @return The current timestamp in milliseconds.
     */
    Time now() const noexcept { return Time{uv_now(loop.get())}; }

    /**
     * @brief Updates the event loop’s concept of _now_.
     *
     * The current time is cached at the start of the event loop tick in order
     * to reduce the number of time-related system calls.<br/>
     * You won’t normally need to call this function unless you have callbacks
     * that block the event loop for longer periods of time, where _longer_ is
     * somewhat subjective but probably on the order of a millisecond or more.
     */
    void update() const noexcept { return uv_update_time(loop.get()); }

    /**
     * @brief Walks the list of handles.
     *
     * The callback will be executed once for each handle that is still active.
     *
     * @param callback A function to be invoked once for each active handle.
     */
    void walk(std::function<void(BaseHandle &)> callback)
    {
        // remember: non-capturing lambdas decay to pointers to functions
        uv_walk(
            loop.get(),
            [](uv_handle_t *handle, void *func) {
                BaseHandle &ref = *static_cast<BaseHandle *>(handle->data);
                std::function<void(BaseHandle &)> &f =
                    *static_cast<std::function<void(BaseHandle &)> *>(func);
                f(ref);
            },
            &callback);
    }

private:
    std::unique_ptr<uv_loop_t, Deleter> loop;
};

} // namespace uvw
