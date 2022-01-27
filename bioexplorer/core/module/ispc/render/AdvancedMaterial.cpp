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

#include "AdvancedMaterial.h"
#include "AdvancedMaterial_ispc.h"

#include <ospray/SDK/common/Data.h>

namespace bioexplorer
{
namespace rendering
{
void AdvancedMaterial::commit()
{
    if (ispcEquivalent == nullptr)
        ispcEquivalent = ispc::AdvancedMaterial_create(this);

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
        MATERIAL_PROPERTY_SHADING_MODE,
        static_cast<int>(MaterialShadingMode::undefined_shading_mode)));

    // User parameter
    userParameter = getParam1f(MATERIAL_PROPERTY_USER_PARAMETER, 1.f);

    // Chameleon mode
    chameleonMode = static_cast<MaterialChameleonMode>(getParam1i(
        MATERIAL_PROPERTY_CHAMELEON_MODE,
        static_cast<int>(MaterialChameleonMode::undefined_chameleon_mode)));

    // Model Id
    nodeId = getParam1i(MATERIAL_PROPERTY_NODE_ID, 0);

    // Cast simulation data
    castSimulationData = getParam(MATERIAL_PROPERTY_CAST_SIMULATION_DATA, 1);

    ispc::AdvancedMaterial_set(
        getIE(), map_d ? map_d->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_d, d,
        map_Refraction ? map_Refraction->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Refraction, refraction,
        map_Reflection ? map_Reflection->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Reflection, reflection,
        map_a ? map_a->getIE() : nullptr, (const ispc::AffineSpace2f&)xform_a,
        a, glossiness, map_Kd ? map_Kd->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Kd, (ispc::vec3f&)Kd,
        map_Ks ? map_Ks->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Ks, (ispc::vec3f&)Ks,
        map_Ns ? map_Ns->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Ns, Ns,
        map_Bump ? map_Bump->getIE() : nullptr,
        (const ispc::AffineSpace2f&)xform_Bump,
        (const ispc::LinearSpace2f&)rot_Bump,
        (const ispc::MaterialShadingMode&)shadingMode, userParameter,
        (const ispc::MaterialChameleonMode&)chameleonMode, nodeId,
        castSimulationData);
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
OSP_REGISTER_MATERIAL(bio_explorer, AdvancedMaterial, default);
OSP_REGISTER_MATERIAL(bio_explorer_fields, AdvancedMaterial, default);
OSP_REGISTER_MATERIAL(bio_explorer_density, AdvancedMaterial, default);
OSP_REGISTER_MATERIAL(bio_explorer_path_tracing, AdvancedMaterial, default);
#endif
} // namespace rendering
} // namespace bioexplorer
