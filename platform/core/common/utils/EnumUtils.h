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

#include <string>
#include <tuple>
#include <vector>

namespace core
{
/*
 * This file contains helper functions for converting strings to and from enums.
 * To use the helper functions define the method 'enumMap' with a specialization
 * for your enum type.
 */

/* Returns a mapping from a name to an enum type. */
template <typename EnumT>
std::vector<std::pair<std::string, EnumT>> enumMap();

/* Returns all names for given enum type 'EnumT' */
template <typename EnumT>
inline std::vector<std::string> enumNames()
{
    std::vector<std::string> v;
    for (const auto& p : enumMap<EnumT>())
        v.push_back(p.first);
    return v;
}

/* Convert a string to an enum. */
template <typename EnumT>
inline EnumT stringToEnum(const std::string& v)
{
    for (const auto& p : enumMap<EnumT>())
        if (p.first == v)
            return p.second;

    throw std::runtime_error("Could not match enum '" + v + "'");
    return static_cast<EnumT>(0);
}

/* Convert an enum to a string. */
template <typename EnumT>
inline std::string enumToString(const EnumT v)
{
    for (const auto& p : enumMap<EnumT>())
        if (p.second == v)
            return p.first;

    throw std::runtime_error("Could not match enum");
    return "Invalid";
}
} // namespace core
