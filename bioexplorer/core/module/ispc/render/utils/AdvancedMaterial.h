/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#pragma once

#include <plugin/common/CommonTypes.h>

#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/texture/Texture2D.h>

namespace bioexplorer
{
namespace rendering
{
using namespace ospray;

typedef vec3f Color;

struct AdvancedMaterial : public Material
{
    /*! opacity: 0 (transparent), 1 (opaque) */
    Texture2D* map_d;
    affine2f xform_d;
    double d{1.0};

    /*! refraction index */
    Texture2D* map_Refraction;
    affine2f xform_Refraction;
    double refraction{1.0};

    /*! reflection index */
    Texture2D* map_Reflection;
    affine2f xform_Reflection;
    double reflection{0.0};

    /*! radiance: 0 (none), 1 (full) */
    Texture2D* map_a;
    affine2f xform_a;
    double a{0.0};

    /*! diffuse  reflectance: 0 (none), 1 (full) */
    Texture2D* map_Kd;
    affine2f xform_Kd;
    Color Kd;

    /*! specular reflectance: 0 (none), 1 (full) */
    Texture2D* map_Ks;
    affine2f xform_Ks;
    Color Ks;

    /*! specular exponent: 0 (diffuse), infinity (specular) */
    Texture2D* map_Ns;
    affine2f xform_Ns;
    double Ns;

    /*! Glossiness: 0 (none), 1 (full) */
    double glossiness{1.0};

    /*! bump map */
    Texture2D* map_Bump;
    affine2f xform_Bump;
    linear2f rot_Bump;

    /*! Shading mode (none, diffuse, electron, etc) */
    MaterialShadingMode shadingMode;

    /*! User parameter */
    double userParameter{1.0};

    /*! Model ID */
    uint32 nodeId;

    /*! Takes the color from surrounding invisible geometry */
    MaterialChameleonMode chameleonMode;

    /*! Determines if shading should include user data */
    bool castUserData{false};

    /*! defines the type of clipping for the geometry attached to the material
     */
    MaterialClippingMode clippingMode;

    std::string toString() const final { return "default_material"; }
    void commit() final;
};
} // namespace rendering
} // namespace bioexplorer
