/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "Utils.h"

#include <platform/core/common/Properties.h>
#include <platform/core/engineapi/Camera.h>

namespace bioexplorer
{
namespace mediamaker
{
using namespace core;

CameraKeyFrame cameraDefinitionToKeyFrame(const CameraDefinition &cd)
{
    CameraKeyFrame keyFrame;
    keyFrame.origin = {cd.origin[0], cd.origin[1], cd.origin[2]};
    keyFrame.direction = {cd.direction[0], cd.direction[1], cd.direction[2]};
    keyFrame.up = {cd.up[0], cd.up[1], cd.up[2]};
    keyFrame.apertureRadius = cd.apertureRadius;
    keyFrame.focalDistance = cd.focalDistance;
    keyFrame.interpupillaryDistance = cd.interpupillaryDistance;
    return keyFrame;
}

CameraDefinition keyFrameToCameraDefinition(const CameraKeyFrame &kf)
{
    CameraDefinition cd;
    cd.origin = {kf.origin.x, kf.origin.y, kf.origin.z};
    cd.direction = {kf.direction.x, kf.direction.y, kf.direction.z};
    cd.up = {kf.up.x, kf.up.y, kf.up.z};
    cd.apertureRadius = kf.apertureRadius;
    cd.focalDistance = kf.focalDistance;
    cd.interpupillaryDistance = kf.interpupillaryDistance;
    return cd;
}

void setCamera(const CameraKeyFrame &keyFrame, Camera &camera, const bool triggerCallback)
{
    camera.setPosition(keyFrame.origin);
    camera.setTarget(keyFrame.origin + keyFrame.direction);
    // Orientation
    const auto q = glm::inverse(glm::lookAt(keyFrame.origin, keyFrame.origin + keyFrame.direction,
                                            keyFrame.up)); // Not quite sure why this
                                                           // should be inverted?!?
    camera.setOrientation(q);

    // Aperture
    if (camera.hasProperty(CAMERA_PROPERTY_APERTURE_RADIUS.name))
        camera.updateProperty(CAMERA_PROPERTY_APERTURE_RADIUS.name, keyFrame.apertureRadius);

    // Focus distance
    if (camera.hasProperty(CAMERA_PROPERTY_FOCAL_DISTANCE.name))
        camera.updateProperty(CAMERA_PROPERTY_FOCAL_DISTANCE.name, keyFrame.focalDistance);

    // Stereo
    if (camera.hasProperty(CAMERA_PROPERTY_STEREO.name))
        camera.updateProperty(CAMERA_PROPERTY_STEREO.name, keyFrame.interpupillaryDistance != 0.0);
    if (camera.hasProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name))
        camera.updateProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name, keyFrame.interpupillaryDistance);

    camera.markModified(triggerCallback);
}

CameraKeyFrame getCameraKeyFrame(Camera &camera)
{
    CameraKeyFrame keyFrame;
    const auto &p = camera.getPosition();
    keyFrame.origin = {p.x, p.y, p.z};
    const auto d = glm::rotate(camera.getOrientation(), core::Vector3d(0., 0., -1.));
    keyFrame.direction = {d.x, d.y, d.z};
    const auto u = glm::rotate(camera.getOrientation(), core::Vector3d(0., 1., 0.));
    keyFrame.up = {u.x, u.y, u.z};
    if (camera.hasProperty(CAMERA_PROPERTY_APERTURE_RADIUS.name))
        keyFrame.apertureRadius = camera.getProperty<double>(CAMERA_PROPERTY_APERTURE_RADIUS.name);
    if (camera.hasProperty(CAMERA_PROPERTY_FOCAL_DISTANCE.name))
        keyFrame.focalDistance = camera.getProperty<double>(CAMERA_PROPERTY_FOCAL_DISTANCE.name);
    if (camera.hasProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name))
        keyFrame.interpupillaryDistance = camera.getProperty<double>(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name);
    return keyFrame;
}
} // namespace mediamaker
} // namespace bioexplorer