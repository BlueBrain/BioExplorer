/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#include "DeflectParameters.h"
#include <deflect/deflect.h>
#include <platform/core/common/utils/EnumUtils.h>

#define deflectPixelOp deflectPixelOp
#define TEXTIFY(A) #A

namespace core
{
template <>
inline std::vector<std::pair<std::string, deflect::ChromaSubsampling>> enumMap()
{
    return {{"yuv444", deflect::ChromaSubsampling::YUV444},
            {"yuv422", deflect::ChromaSubsampling::YUV422},
            {"yuv420", deflect::ChromaSubsampling::YUV420}};
}
namespace utils
{
/**
 * Decode the framebuffer name to know the eye pass. Defined by plugins that
 * follow our convention, and by default in Core.cpp
 */
inline deflect::View getView(const std::string& name)
{
    if (name.length() == 2)
    {
        if (name.at(1) == 'L')
            return deflect::View::left_eye;
        if (name.at(1) == 'R')
            return deflect::View::right_eye;
        return deflect::View::mono;
    }
    return deflect::View::mono;
}

/**
 * Decode the framebuffer name to know the surface. Defined by plugins that
 * follow our convention, and by default in Core.cpp
 */
inline uint8_t getChannel(const std::string& name)
{
    if (name.length() == 2)
        return std::stoi(&name.at(0));
    return 0;
}

inline bool needsReset(const deflect::Observer& stream, const DeflectParameters& params)
{
    const bool changed = stream.getId() != params.getId() || stream.getPort() != params.getPort() ||
                         stream.getHost() != params.getHostname();

    return changed || !stream.isConnected() || !params.getEnabled();
}
} // namespace utils
} // namespace core
