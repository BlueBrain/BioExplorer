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

#include "Utils.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/utils/FileSystem.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <set>
#include <sstream>
#include <string>

namespace core
{
strings parseFolder(const std::string& folder, const strings& filters)
{
    strings files;
    fs::directory_iterator endIter;
    if (fs::is_directory(folder))
    {
        for (fs::directory_iterator dirIter(folder); dirIter != endIter; ++dirIter)
        {
            if (fs::is_regular_file(dirIter->status()))
            {
                const auto filename = dirIter->path().c_str();
                if (filters.empty())
                    files.push_back(filename);
                else
                {
                    const auto& fileExtension = dirIter->path().extension();
                    const auto found = std::find(filters.begin(), filters.end(), fileExtension);
                    if (found != filters.end())
                        files.push_back(filename);
                }
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string extractExtension(const std::string& filename)
{
    auto extension = fs::path(filename).extension().string();
    if (!extension.empty())
        extension = extension.erase(0, 1);

    return extension;
}

Vector4f getBezierPoint(const Vector4fs& controlPoints, const float t)
{
    const auto clampt = glm::clamp(t, 0.f, 1.f);
    const size_t n = controlPoints.size();

    Vector4fs tempPoints = controlPoints;
    for (uint64_t k = 1; k < n; ++k)
    {
        for (uint64_t i = 0; i < n - k; ++i)
        {
            tempPoints[i].x = (1 - clampt) * tempPoints[i].x + clampt * tempPoints[i + 1].x;
            tempPoints[i].y = (1 - clampt) * tempPoints[i].y + clampt * tempPoints[i + 1].y;
            tempPoints[i].z = (1 - clampt) * tempPoints[i].z + clampt * tempPoints[i + 1].z;
            tempPoints[i].w = (1 - clampt) * tempPoints[i].w + clampt * tempPoints[i + 1].w;
        }
    }
    return tempPoints[0];
}

struct RGBColor
{
    int r, g, b;
};

Vector3f hsvToRgb(float h, float s, float v)
{
    int i = h * 6;
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    switch (i % 6)
    {
    case 0:
        return {v, t, p};
    case 1:
        return {q, v, p};
    case 2:
        return {p, v, t};
    case 3:
        return {p, q, v};
    case 4:
        return {t, p, v};
    case 5:
        return {v, p, q};
    default:
        return {0, 0, 0}; // Should not reach here
    }
}

Vector3fs getRainbowColormap(const uint32_t colormapSize)
{
    Vector3fs colormap;
    for (uint32_t i = 0; i < colormapSize; ++i)
    {
        const float hue = static_cast<float>(i) / colormapSize;
        colormap.push_back(hsvToRgb(hue, 1.0f, 1.0f));
    }

    return colormap;
}

template <typename To, typename From>
To lexical_cast(const From& from)
{
    To to;
    std::from_chars(from.data(), from.data() + from.size(), to);
    return to;
}

} // namespace core
