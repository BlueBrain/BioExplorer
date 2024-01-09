/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/transferFunction/TransferFunction.h>
#include <platform/core/engineapi/Field.h>

#include "OptiXModel.h"
#include "OptiXTypes.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

namespace core
{
namespace engine
{
namespace optix
{
class OptiXField : public Field
{
public:
    /** @copydoc Volume::Volume */
    OptiXField(const Vector3ui& dimensions, const Vector3f& spacing, const VolumeParameters& params);

    /** @copydoc OctreeVolume::setOctree */
    void setOctree(const Vector3f& offset, const uint32_ts& indices, const floats& values,
                   const OctreeDataType dataType) final;
};
} // namespace optix
} // namespace engine
} // namespace core