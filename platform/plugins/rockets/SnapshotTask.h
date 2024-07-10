/*
 * Copyright (c) 2015-2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel.Nachbaur@epfl.ch
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include "ImageGenerator.h"

#include <platform/core/common/Properties.h>
#include <platform/core/common/tasks/Task.h>
#include <platform/core/common/utils/StringUtils.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Renderer.h>
#include <platform/core/engineapi/Scene.h>

#include <platform/core/parameters/ParametersManager.h>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imageio.h>

#include <fstream>

OIIO_NAMESPACE_USING

namespace core
{
struct SnapshotParams
{
    std::unique_ptr<AnimationParameters> animParams;
    std::unique_ptr<GeometryParameters> geometryParams;
    std::unique_ptr<VolumeParameters> volumeParams;
    std::unique_ptr<FieldParameters> fieldParams;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<Camera> camera;
    int samplesPerPixel{1};
    Vector2ui size;
    size_t quality{100};
    std::string format; // FreeImage formats apply
    std::string name;
    std::string filePath;
};

/**
 * A functor for snapshot rendering and conversion to a base64-encoded image for
 * the web client.
 */
class SnapshotFunctor : public TaskFunctor
{
public:
    SnapshotFunctor(Engine& engine, SnapshotParams&& params, ImageGenerator& imageGenerator)
        : _params(std::move(params))
        , _camera(engine.createCamera())
        , _imageGenerator(imageGenerator)
        , _engine(engine)
    {
        auto& applicationParameters = engine.getParametersManager().getApplicationParameters();
        const auto& engineName = applicationParameters.getEngine();
        if (engineName == ENGINE_OPTIX_6)
            CORE_THROW("Snapshot are currently not supported by the " + engineName + " engine");

        const auto& parametersManager = engine.getParametersManager();
        if (_params.animParams == nullptr)
        {
            _params.animParams = std::make_unique<AnimationParameters>(parametersManager.getAnimationParameters());
        }

        if (_params.geometryParams == nullptr)
        {
            _params.geometryParams = std::make_unique<GeometryParameters>(parametersManager.getGeometryParameters());
        }

        if (_params.volumeParams == nullptr)
        {
            _params.volumeParams = std::make_unique<VolumeParameters>(parametersManager.getVolumeParameters());
        }

        if (_params.fieldParams == nullptr)
        {
            _params.fieldParams = std::make_unique<FieldParameters>(parametersManager.getFieldParameters());
        }

        _scene = engine.createScene(*_params.animParams, *_params.geometryParams, *_params.volumeParams,
                                    *_params.fieldParams);

        if (_params.camera)
        {
            *_camera = *_params.camera;
            _camera->setCurrentType(engine.getCamera().getCurrentType());
            _camera->clonePropertiesFrom(engine.getCamera());
        }
        else
            *_camera = engine.getCamera();

        _renderer = engine.createRenderer(*_params.animParams, parametersManager.getRenderingParameters());

        const auto& renderer = engine.getRenderer();
        _renderer->setCurrentType(renderer.getCurrentType());
        _renderer->clonePropertiesFrom(renderer);

        _scene->copyFrom(engine.getScene());
    }

    ImageGenerator::ImageBase64 operator()()
    {
        _scene->commit();

        _camera->updateProperty(CAMERA_PROPERTY_ASPECT_RATIO.name, double(_params.size.x) / _params.size.y);
        _camera->commit();

        _renderer->setSamplesPerPixel(1);
        _renderer->setSubsampling(1);
        _renderer->setCamera(_camera);
        _renderer->setEngine(&_engine);
        _renderer->commit();

        std::stringstream msg;
        msg << "Render snapshot";
        if (!_params.name.empty())
            msg << " " << string_utils::shortenString(_params.name);
        msg << " ...";

        const auto isStereo = _camera->hasProperty(CAMERA_PROPERTY_STEREO.name) &&
                              _camera->getProperty<bool>(CAMERA_PROPERTY_STEREO.name);
        const auto names = isStereo ? strings{"0L", "0R"} : strings{DEFAULT};
        std::vector<FrameBufferPtr> frameBuffers;
        for (const auto& name : names)
            frameBuffers.push_back(_engine.createFrameBuffer(name, _params.size, FrameBufferFormat::rgba_i8));

        while (frameBuffers[0]->numAccumFrames() != size_t(_params.samplesPerPixel))
        {
            for (auto frameBuffer : frameBuffers)
            {
                _camera->setBufferTarget(frameBuffer->getName());
                _camera->markModified(false);
                _camera->commit();
                _camera->resetModified();
                _renderer->render(frameBuffer);
                frameBuffer->incrementAccumFrames();
            }

            progress(msg.str(), 1.f / frameBuffers[0]->numAccumFrames(),
                     float(frameBuffers[0]->numAccumFrames()) / _params.samplesPerPixel);
        }

        if (!_params.filePath.empty() && frameBuffers.size() == 1)
        {
            auto& fb = *frameBuffers[0];
            _writeToDisk(fb);

            return ImageGenerator::ImageBase64();
        }
        else
            return _imageGenerator.createImage(frameBuffers, _params.format, _params.quality);
    }

private:
    void writeBufferToFile(const std::vector<unsigned char>& buffer, const std::string& path)
    {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to create " + path);
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        file.close();
    }

    void _writeToDisk(FrameBuffer& fb)
    {
        auto image = fb.getImage();

        // Determine the output format
        std::string format = _params.format;
        if (format != "jpg" && format != "png" && format != "tiff")
            CORE_THROW("Unknown format: " + format);

        int quality = _params.quality;

        // Prepare the file path
        const std::string path = _params.filePath + "." + format;

        // Set up ImageSpec for output image
        ImageSpec spec = image.spec();
        if (format == "jpg")
            spec.attribute("CompressionQuality", quality);

        // Create an output buffer and write image to memory
        std::vector<unsigned char> buffer(spec.image_bytes());

        auto out = ImageOutput::create(path);
        if (!out)
            CORE_THROW("Could not create image output");

        out->open(path, spec);
        out->write_image(TypeDesc::UINT8, image.localpixels());
        out->close();

        // Write the buffer content to the file
        writeBufferToFile(buffer, path);
    }

    SnapshotParams _params;
    FrameBufferPtr _frameBuffer;
    CameraPtr _camera;
    RendererPtr _renderer;
    ScenePtr _scene;
    ImageGenerator& _imageGenerator;
    Engine& _engine;
};
} // namespace core
