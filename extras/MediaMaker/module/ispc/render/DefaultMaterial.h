/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/texture/Texture2D.h>

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
typedef ospray::vec3f Color;

struct DefaultMaterial : public ospray::Material
{
    /*! opacity: 0 (transparent), 1 (opaque) */
    ospray::Texture2D* map_d;
    double d;

    /*! refraction index */
    ospray::Texture2D* map_Refraction;
    double refraction;

    /*! reflection index */
    ospray::Texture2D* map_Reflection;
    double reflection;

    /*! radiance: 0 (none), 1 (full) */
    ospray::Texture2D* map_a;
    double a;

    /*! diffuse  reflectance: 0 (none), 1 (full) */
    ospray::Texture2D* map_Kd;
    Color Kd;

    /*! specular reflectance: 0 (none), 1 (full) */
    ospray::Texture2D* map_Ks;
    Color Ks;

    /*! specular exponent: 0 (diffuse), infinity (specular) */
    ospray::Texture2D* map_Ns;
    double Ns;

    /*! Glossiness: 0 (none), 1 (full) */
    double glossiness;

    /*! bump map */
    ospray::Texture2D* map_Bump;

    std::string toString() const override
    {
        return "mediamaker::DefaultMaterial";
    }

    void commit() override;
};
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer