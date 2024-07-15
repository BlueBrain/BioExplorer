/*
    Copyright 2019 - 2024 Blue Brain Project / EPFL

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

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <platform/core/common/Timer.h>
#include <platform/core/common/Types.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace core
{
class Picture
{
public:
    AVFrame *frame{nullptr};

    int init(enum AVPixelFormat pix_fmt, int width, int height)
    {
        frame = av_frame_alloc();
        frame->format = pix_fmt;
        frame->width = width;
        frame->height = height;
        return av_frame_get_buffer(frame, 32);
    }

    ~Picture()
    {
        if (frame)
            av_frame_free(&frame);
    }
};

template <typename T, size_t S = 2>
class MTQueue
{
public:
    explicit MTQueue(const size_t maxSize = S)
        : _maxSize(maxSize)
    {
    }

    void clear()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        std::queue<T>().swap(_queue);
        _condition.notify_all();
    }

    void push(const T &element)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [&] { return _queue.size() < _maxSize; });
        _queue.push(element);
        _condition.notify_all();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [&] { return !_queue.empty(); });

        T element = _queue.front();
        _queue.pop();
        _condition.notify_all();
        return element;
    }
    size_t size() const
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size();
    }

private:
    std::queue<T> _queue;
    mutable std::mutex _mutex;
    mutable std::condition_variable _condition;
    const size_t _maxSize;
};

class Encoder
{
public:
    using DataFunc = std::function<void(const char *data, size_t size)>;

    Encoder(const int width, const int height, const int fps, const int64_t kbps, const DataFunc &dataFunc);
    ~Encoder();

    void encode(FrameBuffer &fb);

    DataFunc _dataFunc;
    const int width;
    const int height;
    const int64_t kbps;

private:
    const int _fps;
    AVFormatContext *formatContext{nullptr};
    AVStream *stream{nullptr};

    AVCodecContext *codecContext{nullptr};
    AVCodec *codec{nullptr};

    SwsContext *sws_context{nullptr};
    Picture picture;

    int64_t _frameNumber{0};

    const bool _async = true;
    std::thread _thread;
    std::atomic_bool _running{true};

    struct Image
    {
        int width{0};
        int height{0};
        std::vector<uint8_t> data;
        bool empty() const { return width == 0 || height == 0; }
        void clear() { width = height = 0; }
    } _image[2];

    MTQueue<int> _queue;
    int _currentImage{0};

    void _runAsync();
    void _encode();
    void _toPicture(const uint8_t *const data, const int width, const int height);

    Timer _timer;
    float _leftover{0.f};
};
} // namespace core
