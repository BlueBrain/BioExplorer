/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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
typedef ospray::vec3f Color;

struct AdvancedMaterial : public ospray::Material
{
    /*! opacity: 0 (transparent), 1 (opaque) */
    ospray::Texture2D* map_d;
    ospray::affine2f xform_d;
    float d;

    /*! refraction index */
    ospray::Texture2D* map_Refraction;
    ospray::affine2f xform_Refraction;
    float refraction;

    /*! reflection index */
    ospray::Texture2D* map_Reflection;
    ospray::affine2f xform_Reflection;
    float reflection;

    /*! radiance: 0 (none), 1 (full) */
    ospray::Texture2D* map_a;
    ospray::affine2f xform_a;
    float a;

    /*! diffuse  reflectance: 0 (none), 1 (full) */
    ospray::Texture2D* map_Kd;
    ospray::affine2f xform_Kd;
    Color Kd;

    /*! specular reflectance: 0 (none), 1 (full) */
    ospray::Texture2D* map_Ks;
    ospray::affine2f xform_Ks;
    Color Ks;

    /*! specular exponent: 0 (diffuse), infinity (specular) */
    ospray::Texture2D* map_Ns;
    ospray::affine2f xform_Ns;
    float Ns;

    /*! Glossiness: 0 (none), 1 (full) */
    float glossiness;

    /*! bump map */
    ospray::Texture2D* map_Bump;
    ospray::affine2f xform_Bump;
    ospray::linear2f rot_Bump;

    /*! Shading mode (none, diffuse, electron, etc) */
    MaterialShadingMode shadingMode;

    /*! User parameter */
    float userParameter;

    /*! Geometry visibily */
    bool visible{true};

    /*! Takes the color from surrounding invisible geometry */
    bool chameleon{false};

    std::string toString() const final { return "default_material"; }
    void commit() final;
};
} // namespace bioexplorer
