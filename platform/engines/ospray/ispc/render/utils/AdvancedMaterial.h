/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/Properties.h>

#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/texture/Texture2D.h>

namespace core
{
namespace engine
{
namespace ospray
{
typedef ::ospray::vec3f Color;

struct AdvancedMaterial : public ::ospray::Material
{
    /*! opacity: 0 (transparent), 1 (opaque) */
    ::ospray::Texture2D* map_d;
    float d;

    /*! refraction index */
    ::ospray::Texture2D* map_Refraction;
    float refraction;

    /*! reflection index */
    ::ospray::Texture2D* map_Reflection;
    float reflection;

    /*! radiance: 0 (none), 1 (full) */
    ::ospray::Texture2D* map_a;
    float a;

    /*! diffuse  reflectance: 0 (none), 1 (full) */
    ::ospray::Texture2D* map_Kd;
    Color Kd;

    /*! specular reflectance: 0 (none), 1 (full) */
    ::ospray::Texture2D* map_Ks;
    Color Ks;

    /*! specular exponent: 0 (diffuse), infinity (specular) */
    ::ospray::Texture2D* map_Ns;
    float Ns;

    /*! Glossiness: 0 (none), 1 (full) */
    float glossiness;

    /*! bump map */
    ::ospray::Texture2D* map_Bump;

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

    std::string toString() const override { return DEFAULT; }

    void commit() override;
};
} // namespace ospray
} // namespace engine
} // namespace core