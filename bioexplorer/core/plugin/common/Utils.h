/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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
void getCubicPosition(const float size, const Vector3f& position,
                      const size_t randomPositionSeed,
                      const float randomPositionStength,
                      const size_t randomOrientationSeed,
                      const float randomOrientationStength, Vector3f& pos,
                      Vector3f& dir);
/**
 * @brief sinusoide
 * @param x
 * @param z
 * @return
 */
float sinusoide(const float x, const float z);

/**
 * @brief getSinosoidalPosition
 * @param size
 * @param height
 * @param randomizationType
 * @param randomSeed
 * @param position
 * @param pos
 * @param dir
 */
void getSinosoidalPosition(const float size, const float amplitude,
                           const PositionRandomizationType randomizationType,
                           const size_t randomPositionSeed,
                           const float randomPositionStrengh,
                           const size_t randomOrientationSeed,
                           const float randomOrientationStrengh,
                           const Vector3f& position, Vector3f& pos,
                           Vector3f& dir);

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

void getSphericalToPlanarPosition(
    const size_t rnd, const float assemblyRadius, const float height,
    const PositionRandomizationType randomizationType, const size_t randomSeed,
    const size_t occurence, const size_t occurences, const Vector3f& position,
    const float morphingStep, Vector3f& pos, Vector3f& dir);

void setTransferFunction(brayns::TransferFunction& tf);

Vector4fs getClippingPlanes(const Scene& scene);

} // namespace bioexplorer
