/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_UTILS_H
#define BIOEXPLORER_UTILS_H

#include <brayns/common/types.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
using namespace brayns;

/**
 * @brief ltrim
 * @param s
 * @return
 */
std::string& ltrim(std::string& s);

/**
 * @brief rtrim
 * @param s
 * @return
 */
std::string& rtrim(std::string& s);

/**
 * @brief trim
 * @param s
 * @return
 */
std::string& trim(std::string& s);

/**
 * @brief isClipped
 * @param position
 * @param clippingPlanes
 * @return
 */
bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes);

/**
 * @brief getSphericalPosition
 * @param rnd
 * @param assemblyRadius
 * @param randomizationType
 * @param randomSeed
 * @param occurence
 * @param occurences
 * @param position
 * @param pos
 * @param dir
 */
void getSphericalPosition(const size_t rnd, const float assemblyRadius,
                          const float height,
                          const PositionRandomizationType randomizationType,
                          const size_t randomSeed, const size_t occurence,
                          const size_t occurences, const Vector3f& position,
                          Vector3f& pos, Vector3f& dir);

/**
 * @brief getPlanarPosition
 * @param assemblyRadius
 * @param randomizationType
 * @param randomSeed
 * @param position
 * @param pos
 * @param dir
 */
void getPlanarPosition(const float assemblyRadius,
                       const PositionRandomizationType randomizationType,
                       const size_t randomSeed, const Vector3f& position,
                       Vector3f& pos, Vector3f& dir);

/**
 * @brief getCubicPosition
 * @param assemblyRadius
 * @param position
 * @param pos
 * @param dir
 */
void getCubicPosition(const float assemblyRadius, const Vector3f& position,
                      Vector3f& pos, Vector3f& dir);
/**
 * @brief sinusoide
 * @param x
 * @param z
 * @return
 */
float sinusoide(const float x, const float z);

/**
 * @brief getSinosoidalPosition
 * @param assemblyRadius
 * @param height
 * @param randomizationType
 * @param randomSeed
 * @param position
 * @param pos
 * @param dir
 */
void getSinosoidalPosition(const float size, const float height,
                           const PositionRandomizationType randomizationType,
                           const size_t randomSeed, const Vector3f& position,
                           Vector3f& pos, Vector3f& dir);

/**
 * @brief getFanPosition
 * @param rnd
 * @param assemblyRadius
 * @param randomizationType
 * @param randomSeed
 * @param occurence
 * @param occurences
 * @param position
 * @param pos
 * @param dir
 */
void getFanPosition(const size_t rnd, const float assemblyRadius,
                    const PositionRandomizationType randomizationType,
                    const size_t randomSeed, const size_t occurence,
                    const size_t occurences, const Vector3f& position,
                    Vector3f& pos, Vector3f& dir);

/**
 * @brief getBezierPosition
 * @param points
 * @param assemblyRadius
 * @param t
 * @param pos
 * @param dir
 */
void getBezierPosition(const Vector3fs& points, const float assemblyRadius,
                       const float t, Vector3f& pos, Vector3f& dir);

} // namespace bioexplorer

#endif // BIOEXPLORER_UTILS_H
