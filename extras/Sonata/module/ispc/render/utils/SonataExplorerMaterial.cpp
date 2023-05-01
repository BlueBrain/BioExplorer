/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "SonataExplorerMaterial.h"
#include "SonataExplorerMaterial_ispc.h"

#include <common/Types.h>
#include <ospray/SDK/common/Data.h>

namespace sonataexplorer
{
void SonataExplorerMaterial::commit()
{
    if (ispcEquivalent == nullptr)
        ispcEquivalent = ispc::SonataExplorerMaterial_create(this);

    // Opacity
    map_d = (ospray::Texture2D*)getParamObject("map_d", nullptr);
    xform_d = getTextureTransform("map_d");
    d = getParam1f("d", 1.f);

    // Diffuse color
    Kd = getParam3f("kd", ospray::vec3f(.8f));
    map_Kd = (ospray::Texture2D*)getParamObject("map_kd", nullptr);
    xform_Kd = getTextureTransform("map_kd");

    // Specular color
    Ks = getParam3f("ks", ospray::vec3f(0.f));
    map_Ks = (ospray::Texture2D*)getParamObject("map_ks", nullptr);
    xform_Ks = getTextureTransform("map_ks");

    // Specular exponent
    Ns = getParam1f("ns", 10.f);
    map_Ns = (ospray::Texture2D*)getParamObject("map_ns", nullptr);
    xform_Ns = getTextureTransform("map_ns");

    // Bump mapping
    map_Bump = (ospray::Texture2D*)getParamObject("map_bump", nullptr);
    xform_Bump = getTextureTransform("map_bump");
    rot_Bump = xform_Bump.l.orthogonal().transposed();

    // Refraction mapping
    refraction = getParam1f("refraction", 0.f);
    xform_Refraction = getTextureTransform("map_refraction");
    map_Refraction =
        (ospray::Texture2D*)getParamObject("map_refraction", nullptr);

    // Reflection mapping
    reflection = getParam1f("reflection", 0.f);
    xform_Reflection = getTextureTransform("map_reflection");
    map_Reflection =
        (ospray::Texture2D*)getParamObject("map_reflection", nullptr);

    // Light emission mapping
    a = getParam1f("a", 0.f);
    xform_a = getTextureTransform("map_a");
    map_a = (ospray::Texture2D*)getParamObject("map_a", nullptr);

    // Glossiness
    glossiness = getParam1f("glossiness", 1.f);

    // Shading mode
    shadingMode = static_cast<MaterialShadingMode>(getParam1i(
        "shading_mode",
        static_cast<int>(MaterialShadingMode::undefined_shading_mode)));

    // User parameter
    userParameter = getParam1f("user_parameter", 1.f);

    // Cast user data
    castUserData = getParam(MATERIAL_PROPERTY_CAST_USER_DATA.c_str(), 0);

    // Clipping mode
    clippingMode = static_cast<MaterialClippingMode>(
        getParam1i(MATERIAL_PROPERTY_CLIPPING_MODE.c_str(),
                   static_cast<int>(MaterialClippingMode::no_clipping)));

    ispc::SonataExplorerMaterial_set(
        getIE(), map_d ? map_d->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_d, d,
        map_Refraction ? map_Refraction->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Refraction, refraction,
        map_Reflection ? map_Reflection->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Reflection, reflection,
        map_a ? map_a->getIE() : nullptr, (const ispc::AffineSpace2f&)xform_a,
        a, glossiness, castUserData, map_Kd ? map_Kd->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Kd, (ispc::vec3f&)Kd,
        map_Ks ? map_Ks->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Ks, (ispc::vec3f&)Ks,
        map_Ns ? map_Ns->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Ns, Ns,
        map_Bump ? map_Bump->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Bump,
        (const ispc::LinearSpace2f&)rot_Bump,
        (const ispc::MaterialShadingMode&)shadingMode, userParameter,
        (const ispc::MaterialClippingMode&)clippingMode);
}

OSP_REGISTER_MATERIAL(circuit_explorer_basic, SonataExplorerMaterial, default);
OSP_REGISTER_MATERIAL(circuit_explorer_advanced, SonataExplorerMaterial,
                      default);
OSP_REGISTER_MATERIAL(circuit_explorer_voxelized_simulation,
                      SonataExplorerMaterial, default);
OSP_REGISTER_MATERIAL(circuit_explorer_cell_growth, SonataExplorerMaterial,
                      default);
OSP_REGISTER_MATERIAL(circuit_explorer_proximity_detection,
                      SonataExplorerMaterial, default);
} // namespace sonataexplorer
