/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Based on OSPRay implementation
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

#include "AdvancedMaterial.h"
#include "AdvancedMaterial_ispc.h"

#include <platform/core/common/Properties.h>

#include <ospray/SDK/common/Data.h>

namespace core
{
namespace engine
{
namespace ospray
{
void AdvancedMaterial::commit()
{
    if (ispcEquivalent == nullptr)
        ispcEquivalent = ::ispc::AdvancedMaterial_create(this);

    // Opacity
    d = getParam1f(MATERIAL_PROPERTY_OPACITY, 1.f);
    map_d = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_OPACITY, nullptr);
    auto xform_d = getTextureTransform(MATERIAL_PROPERTY_MAP_OPACITY);

    // Diffuse color
    Kd = getParam3f(MATERIAL_PROPERTY_DIFFUSE_COLOR, ::ospray::vec3f(.8f));
    map_Kd = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_DIFFUSE_COLOR, nullptr);
    auto xform_Kd = getTextureTransform(MATERIAL_PROPERTY_MAP_DIFFUSE_COLOR);

    // Specular color
    Ks = getParam3f(MATERIAL_PROPERTY_SPECULAR_COLOR, ::ospray::vec3f(0.f));
    map_Ks = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_SPECULAR_COLOR, nullptr);
    auto xform_Ks = getTextureTransform(MATERIAL_PROPERTY_MAP_SPECULAR_COLOR);

    // Specular exponent
    Ns = getParam1f(MATERIAL_PROPERTY_SPECULAR_INDEX, 10.f);
    map_Ns = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_SPECULAR_INDEX, nullptr);
    auto xform_Ns = getTextureTransform(MATERIAL_PROPERTY_MAP_SPECULAR_INDEX);

    // Bump mapping
    map_Bump = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_BUMP, nullptr);
    auto xform_Bump = getTextureTransform(MATERIAL_PROPERTY_MAP_BUMP);
    auto rot_Bump = xform_Bump.l.orthogonal().transposed();

    // Refraction mapping
    refraction = getParam1f(MATERIAL_PROPERTY_REFRACTION, 0.f);
    map_Refraction = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_REFRACTION, nullptr);
    auto xform_Refraction = getTextureTransform(MATERIAL_PROPERTY_MAP_REFRACTION);

    // Reflection mapping
    reflection = getParam1f(MATERIAL_PROPERTY_REFLECTION, 0.f);
    map_Reflection = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_REFLECTION, nullptr);
    auto xform_Reflection = getTextureTransform(MATERIAL_PROPERTY_MAP_REFLECTION);

    // Light emission mapping
    a = getParam1f(MATERIAL_PROPERTY_EMISSION, 0.f);
    map_a = (::ospray::Texture2D*)getParamObject(MATERIAL_PROPERTY_MAP_EMISSION, nullptr);
    auto xform_a = getTextureTransform(MATERIAL_PROPERTY_MAP_EMISSION);

    // Glossiness
    glossiness = getParam1f(MATERIAL_PROPERTY_GLOSSINESS, 1.f);

    // User parameter
    userParameter = getParam1f(MATERIAL_PROPERTY_USER_PARAMETER, 1.f);

    // Shading mode
    shadingMode = static_cast<MaterialShadingMode>(
        getParam1i(MATERIAL_PROPERTY_SHADING_MODE, MaterialShadingMode::undefined_shading_mode));

    // Cast user data
    castUserData = getParam(MATERIAL_PROPERTY_CAST_USER_DATA, 0);

    // Clipping mode
    clippingMode = static_cast<MaterialClippingMode>(
        getParam1i(MATERIAL_PROPERTY_CLIPPING_MODE, MaterialClippingMode::no_clipping));

    // Node Id
    nodeId = getParam1i(MATERIAL_PROPERTY_NODE_ID, 0);

    // Chameleon mode
    chameleonMode = static_cast<MaterialChameleonMode>(
        getParam1i(MATERIAL_PROPERTY_CHAMELEON_MODE, MaterialChameleonMode::undefined_chameleon_mode));

    ::ispc::AdvancedMaterial_set(
        getIE(), map_d ? map_d->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_d, d,
        map_Refraction ? map_Refraction->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_Refraction, refraction,
        map_Reflection ? map_Reflection->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_Reflection, reflection,
        map_a ? map_a->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_a, a, glossiness,
        map_Kd ? map_Kd->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_Kd, (ispc::vec3f&)Kd,
        map_Ks ? map_Ks->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_Ks, (ispc::vec3f&)Ks,
        map_Ns ? map_Ns->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_Ns, Ns,
        map_Bump ? map_Bump->getIE() : nullptr, (const ::ispc::AffineSpace2f&)xform_Bump,
        (const ::ispc::LinearSpace2f&)rot_Bump, userParameter, (const ::ispc::MaterialShadingMode&)shadingMode,
        castUserData, (const ::ispc::MaterialClippingMode&)clippingMode, nodeId,
        (const ::ispc::MaterialChameleonMode&)chameleonMode);
}

OSP_REGISTER_MATERIAL(basic, AdvancedMaterial, default);
OSP_REGISTER_MATERIAL(advanced, AdvancedMaterial, default);
} // namespace ospray
} // namespace engine
} // namespace core