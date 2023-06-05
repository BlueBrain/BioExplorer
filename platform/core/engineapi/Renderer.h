/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Core <https://github.com/BlueBrain/Core>
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

#include <platform/core/common/Api.h>
#include <platform/core/common/PropertyObject.h>
#include <platform/core/parameters/AnimationParameters.h>
#include <platform/core/parameters/RenderingParameters.h>

namespace core
{
class Renderer : public PropertyObject
{
public:
    struct PickResult
    {
        bool hit{false};
        Vector3d pos;
    };

    /** @name API for engine-specific code */
    //@{
    virtual void render(FrameBufferPtr frameBuffer) = 0;

    /** @return the variance from the previous render(). */
    virtual float getVariance() const { return 0.f; }
    virtual void commit() = 0;
    virtual void setCamera(CameraPtr camera) = 0;
    virtual PickResult pick(const Vector2f& /*pickPos*/) { return PickResult(); }
    //@}

    PLATFORM_API Renderer(const AnimationParameters& animationParameters, const RenderingParameters& renderingParameters);

    void setScene(ScenePtr scene) { _scene = scene; };

protected:
    const AnimationParameters& _animationParameters;
    const RenderingParameters& _renderingParameters;
    ScenePtr _scene;
};
} // namespace core
