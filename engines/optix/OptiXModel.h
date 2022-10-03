/* Copyright (c) 2015-2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "OptiXTypes.h"

#include <brayns/engineapi/Model.h>

#include <map>

namespace brayns
{
class OptiXModel : public Model
{
public:
    OptiXModel(AnimationParameters& animationParameters,
               VolumeParameters& volumeParameters);

    /** @copydoc Model::commit */
    void commitGeometry() final;

    /** @copydoc Model::buildBoundingBox */
    void buildBoundingBox() final;

    /** @copydoc Model::createMaterialImpl */
    virtual MaterialPtr createMaterialImpl(
        const PropertyMap& properties = {}) final;

    /** @copydoc Model::createSharedDataVolume */
    virtual SharedDataVolumePtr createSharedDataVolume(
        const Vector3ui& dimensions, const Vector3f& spacing,
        const DataType type) const final;

    /** @copydoc Model::createBrickedVolume */
    virtual BrickedVolumePtr createBrickedVolume(
        const Vector3ui& dimensions, const Vector3f& spacing,
        const DataType type) const final;

protected:
    void _commitTransferFunctionImpl(const Vector3fs& colors,
                                     const floats& opacities,
                                     const Vector2d valueRange) final;
    void _commitSimulationDataImpl(const float* frameData,
                                   const size_t frameSize) final;

private:
    void _createGas();
    void _commitSpheres(const size_t materialId);
    void _commitCylinders(const size_t materialId);
    void _commitCones(const size_t materialId);
    void _commitMeshes(const size_t materialId);
    void _commitMaterials();
    bool _commitSimulationData();
    bool _commitTransferFunction();

    bool _boundingBoxBuilt = false;
};
} // namespace brayns
