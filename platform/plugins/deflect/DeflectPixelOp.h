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

#include "DeflectParameters.h"

#include <deflect/Stream.h>
#include <map>
#include <ospray/SDK/fb/PixelOp.h>

namespace core
{
/**
 * Implements an ospray pixel op that streams each tile to a Deflect server
 * instance. The tiles are compressed directly on the tile thread and then
 * enqueued for sending.
 *
 * The ospray module to load is called "deflect", and the pixel op name for
 * creating it is "DeflectPixelOp".
 */
class DeflectPixelOp : public ::ospray::PixelOp
{
public:
    struct Instance : public ::ospray::PixelOp::Instance
    {
        Instance(::ospray::FrameBuffer* fb_, DeflectPixelOp& parent);
        ~Instance();

        void beginFrame() final;
        void endFrame() final;
        void postAccum(::ospray::Tile& tile) final;
        std::string toString() const final { return "DeflectPixelOp"; }
        struct PixelsDeleter
        {
            void operator()(unsigned char* pixels) { _mm_free(pixels); }
        };
        using Pixels = std::unique_ptr<unsigned char, PixelsDeleter>;

    private:
        DeflectPixelOp& _parent;
        std::vector<Pixels> _pixels;

        unsigned char* _copyPixels(::ospray::Tile& tile, const ::ospray::vec2i& tileSize);
    };

    /**
     * Updates the underlying deflect stream with the following parameters:
     * - "enabled" (param1i): 1 to enable streaming, 0 to disable streaming,
     *                        1 default
     * - "compression" (param1i): 1 to enable compression, 0 to send raw,
     *                            uncompressed pixels, 1 default
     * - "quality" (param1i): 1 (worst, smallest) - 100 (best, biggest) for JPEG
     *                        quality, 80 default
     */
    void commit() final;

    ::ospray::PixelOp::Instance* createInstance(::ospray::FrameBuffer* fb, PixelOp::Instance* prev) final;

private:
    /** @internal finish pendings sends before closing the stream. */
    void _finish();

    std::unique_ptr<deflect::Stream> _deflectStream;
    std::map<pthread_t, std::shared_future<bool>> _finishFutures;
    std::mutex _mutex;
    DeflectParameters _params;
};
} // namespace core
