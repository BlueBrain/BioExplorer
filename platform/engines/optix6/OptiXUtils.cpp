/*
 * Copyright (c) 2023-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "OptiXUtils.h"
#include "OptiXContext.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/PropertyObject.h>

namespace core
{
namespace engine
{
namespace optix
{
void toOptiXProperties(const PropertyMap& object)
{
    try
    {
        auto context = OptiXContext::get().getOptixContext();
        for (const auto& prop : object.getProperties())
        {
            switch (prop->type)
            {
            case Property::Type::Double:
                context[prop->name]->setFloat(static_cast<float>(prop->get<double>()));
                break;
            case Property::Type::Int:
                context[prop->name]->setInt(prop->get<int32_t>());
                break;
            case Property::Type::Bool:
                // Special case, no bool in OptiX
                context[prop->name]->setUint(prop->get<bool>());
                break;
            case Property::Type::String:
                CORE_WARN("Cannot upload string property to OptiX '" << prop->name << "'");
                break;
            case Property::Type::Vec2d:
            {
                auto v = prop->get<std::array<double, 2>>();
                context[prop->name]->setFloat(static_cast<float>(v[0]), static_cast<float>(v[1]));
                break;
            }
            case Property::Type::Vec2i:
            {
                auto v = prop->get<std::array<int32_t, 2>>();
                context[prop->name]->setInt(v[0], v[1]);
                break;
            }
            case Property::Type::Vec3d:
            {
                auto v = prop->get<std::array<double, 3>>();
                context[prop->name]->setFloat(static_cast<float>(v[0]), static_cast<float>(v[1]),
                                              static_cast<float>(v[2]));
                break;
            }
            case Property::Type::Vec3i:
            {
                auto v = prop->get<std::array<int32_t, 3>>();
                context[prop->name]->setInt(v[0], v[1], v[2]);
                break;
            }
            case Property::Type::Vec4d:
            {
                auto v = prop->get<std::array<double, 4>>();
                context[prop->name]->setFloat(static_cast<float>(v[0]), static_cast<float>(v[1]),
                                              static_cast<float>(v[2]), static_cast<float>(v[3]));
                break;
            }
            }
        }
    }
    catch (const std::exception& e)
    {
        CORE_ERROR("Failed to apply properties for OptiX object" << e.what());
    }
}
} // namespace optix
} // namespace engine
} // namespace core