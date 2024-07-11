/*
    Copyright 2023 - 2024 Blue Brain Project / EPFL

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