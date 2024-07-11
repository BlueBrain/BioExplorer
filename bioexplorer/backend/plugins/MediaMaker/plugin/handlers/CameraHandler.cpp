/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "CameraHandler.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace mediamaker
{
using namespace core;

CameraHandler::CameraHandler(Camera& camera, const CameraKeyFrames& keyFrames, const uint64_t stepsBetweenKeyFrames,
                             const uint64_t numberOfSmoothingSteps)
    : core::AbstractAnimationHandler()
    , _camera(camera)
    , _keyFrames(keyFrames)
    , _stepsBetweenKeyFrames(stepsBetweenKeyFrames)
    , _numberOfSmoothingSteps(numberOfSmoothingSteps)
{
    _frameSize = 0;
    _nbFrames = stepsBetweenKeyFrames * (keyFrames.size() - 1);
    _frameData.resize(_frameSize);
    _dt = 1.f;
    _buildCameraPath();
    _logSimulationInformation();
}

void CameraHandler::_logSimulationInformation()
{
    PLUGIN_INFO("---------------------------------------------------------");
    PLUGIN_INFO("Camera information");
    PLUGIN_INFO("---------------------------");
    PLUGIN_INFO("Start time               : " << 0);
    PLUGIN_INFO("End time                 : " << _nbFrames);
    PLUGIN_INFO("Time interval            : " << _dt);
    PLUGIN_INFO("---------------------------------------------------------");
}

CameraHandler::CameraHandler(const CameraHandler& rhs)
    : core::AbstractAnimationHandler(rhs)
    , _camera(rhs._camera)
{
}

void* CameraHandler::getFrameData(const uint32_t frame)
{
    const auto boundedFrame = _getBoundedFrame(frame);
    if (boundedFrame != _currentFrame)
    {
        setCamera(_smoothedKeyFrames[boundedFrame], _camera, false);
        _currentFrame = boundedFrame;
    }
    return nullptr;
}

core::AbstractSimulationHandlerPtr CameraHandler::clone() const
{
    return std::make_shared<CameraHandler>(*this);
}

void CameraHandler::_buildCameraPath()
{
    Vector3ds origins;
    Vector3ds directions;
    Vector3ds ups;
    doubles aperture_radii;
    doubles focus_distances;
    _smoothedKeyFrames.clear();
    const double numberOfSmoothingSteps = static_cast<double>(_numberOfSmoothingSteps);
    const double stepsBetweenKeyFrames = static_cast<double>(_stepsBetweenKeyFrames);

    for (uint64_t s = 0; s < _keyFrames.size() - 1; ++s)
    {
        const auto& p0 = _keyFrames[s];
        const auto& p1 = _keyFrames[s + 1];

        for (uint64_t i = 0; i < _stepsBetweenKeyFrames; ++i)
        {
            Vector3d origin = {0, 0, 0};
            Vector3d direction = {0, 0, 0};
            Vector3d up = {0, 0, 0};

            origin = p0.origin + (p1.origin - p0.origin) * static_cast<double>(i) / stepsBetweenKeyFrames;
            direction = p0.direction + (p1.direction - p0.direction) * static_cast<double>(i) / stepsBetweenKeyFrames;
            up = p0.up + (p1.up - p0.up) * static_cast<double>(i) / stepsBetweenKeyFrames;

            float aperture_radius = p0.apertureRadius + (p1.apertureRadius - p0.apertureRadius) *
                                                            static_cast<double>(i) / stepsBetweenKeyFrames;
            float focus_distance = p0.focalDistance + (p1.focalDistance - p0.focalDistance) * static_cast<double>(i) /
                                                          stepsBetweenKeyFrames;

            origins.push_back(origin);
            directions.push_back(direction);
            ups.push_back(up);
            aperture_radii.push_back(aperture_radius);
            focus_distances.push_back(focus_distance);
        }
    }

    size_t nb_frames = origins.size();
    for (uint64_t i = 0; i < nb_frames; ++i)
    {
        Vector3d o;
        Vector3d d;
        Vector3d u;
        double aperture_radius = 0.0;
        double focus_distance = 0.0;

        for (int64_t j = 0; j < _numberOfSmoothingSteps; ++j)
        {
            const int64_t index =
                std::max(0l, std::min(static_cast<int64_t>(i) + j - static_cast<int64_t>(_numberOfSmoothingSteps) / 2,
                                      static_cast<int64_t>(nb_frames)));
            o += origins[index];
            d += directions[index];
            u += ups[index];
            aperture_radius += aperture_radii[index];
            focus_distance += focus_distances[index];
        }

        _smoothedKeyFrames.push_back({o / numberOfSmoothingSteps, d / numberOfSmoothingSteps,
                                      u / numberOfSmoothingSteps, aperture_radius / numberOfSmoothingSteps,
                                      focus_distance / numberOfSmoothingSteps});
    }
}

} // namespace mediamaker
} // namespace bioexplorer
