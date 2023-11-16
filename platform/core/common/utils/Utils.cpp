/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#include "Utils.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/utils/FileSystem.h>

#include <algorithm>
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

Vector4f getBezierPoint(const Vector4fs& controlPoints, const double t)
{
    if (t < 0.0 || t > 1.0)
        CORE_THROW("Invalid value with t=" + std::to_string(t) + ". Must be between 0 and 1");
    const uint64_t nbControlPoints = controlPoints.size();
    // 3D points
    Vector3fs points;
    points.reserve(nbControlPoints);
    for (const auto& controlPoint : controlPoints)
        points.push_back({controlPoint.x, controlPoint.y, controlPoint.z});
    for (int64_t i = nbControlPoints - 1; i >= 0; --i)
        for (uint64_t j = 0; j < i; ++j)
            points[j] += t * (points[j + 1] - points[j]);

    // Radius
    const double radius = controlPoints[floor(t * double(nbControlPoints))].w;
    return Vector4f(points[0].x, points[0].y, points[0].z, radius);
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

} // namespace core
