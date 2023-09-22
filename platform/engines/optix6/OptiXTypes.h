/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <memory>

#include <platform/core/common/Types.h>

namespace core
{
namespace engine
{
namespace optix
{
class OptiXCamera;
using OptiXCameraPtr = std::shared_ptr<OptiXCamera>;
class OptiXCameraProgram;
using OptiXCameraProgramPtr = std::shared_ptr<OptiXCameraProgram>;

struct VolumeGeometry
{
    Vector3f dimensions;
    Vector3f position;
    Vector3f spacing;
    float volumeSamplerId;
    float transferFunctionSamplerId;
    Vector2f valueRange;
};
} // namespace optix
} // namespace engine
} // namespace core