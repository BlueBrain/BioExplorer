/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * Based on OSPRay implementation
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

#include <brayns/common/CommonTypes.h>

#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/texture/Texture2D.h>

namespace brayns
{
typedef ospray::vec3f Color;

struct AdvancedMaterial : public ospray::Material
{
    /*! opacity: 0 (transparent), 1 (opaque) */
    ospray::Texture2D* map_d;
    float d;

    /*! refraction index */
    ospray::Texture2D* map_Refraction;
    float refraction;

    /*! reflection index */
    ospray::Texture2D* map_Reflection;
    float reflection;

    /*! radiance: 0 (none), 1 (full) */
    ospray::Texture2D* map_a;
    float a;

    /*! diffuse  reflectance: 0 (none), 1 (full) */
    ospray::Texture2D* map_Kd;
    Color Kd;

    /*! specular reflectance: 0 (none), 1 (full) */
    ospray::Texture2D* map_Ks;
    Color Ks;

    /*! specular exponent: 0 (diffuse), infinity (specular) */
    ospray::Texture2D* map_Ns;
    float Ns;

    /*! Glossiness: 0 (none), 1 (full) */
    float glossiness;

    /*! bump map */
    ospray::Texture2D* map_Bump;

    /*! User defined parameter */
    float userParameter;

    /*! Shading mode */
    MaterialShadingMode shadingMode{MaterialShadingMode::undefined_shading_mode};

    /*! Cast user data on geometry */
    bool castUserData{false};

    /*! Clipping mode applied to geometry */
    MaterialClippingMode clippingMode{MaterialClippingMode::no_clipping};

    /*! Id of the node associated to the material */
    uint32_t nodeId{0};

    /*! Clipping mode applied to geometry */
    MaterialChameleonMode chameleonMode{MaterialChameleonMode::undefined_chameleon_mode};

    std::string toString() const override { return "brayns::AdvancedMaterial"; }

    void commit() override;
};
} // namespace brayns
