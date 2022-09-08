/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/Types.h>

#include <brayns/common/types.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

/**
 * @brief Left trim of a string
 *
 * @param s String to trim
 * @return Trimmed string
 */
std::string& ltrim(std::string& s);

/**
 * @brief Right trim of a string
 *
 * @param s String to trim
 * @return Trimmed string
 */
std::string& rtrim(std::string& s);

/**
 * @brief Left and right trim of a string
 *
 * @param s String to trim
 * @return Trimmed string
 */
std::string& trim(std::string& s);

/**
 * @brief isClipped Determine if a 3d position is inside a volume defined by
 * clipping planes
 *
 * @param position Position to check
 * @param clippingPlanes Clipping planes defining the volume
 * @return True if the position does not belong to the volume, false otherwise
 */
bool isClipped(const Vector3d& position, const Vector4ds& clippingPlanes);

/**
 * @brief Set the default transfer function (Unipolar) to a given model
 *
 * @param model Model to which the transfer function should be set
 */
void setDefaultTransferFunction(Model& model,
                                const Vector2d range = {0.0, 1.0});

/**
 * @brief Get the Clipping Planes from the scene
 *
 * @param scene 3D scene
 * @return Vector4ds List of clipping planes
 */
Vector4ds getClippingPlanes(const Scene& scene);

/**
 * @brief Converts a vector of doubles into a 2D vector
 *
 * @param value Vector of doubles
 * @return Vector2d A 2D vector
 */
Vector2d doublesToVector2d(const doubles& value);

/**
 * @brief Converts a vector of doubles into a 3D vector
 *
 * @param value Vector of doubles
 * @return Vector3d A 3D vector
 */
Vector3d doublesToVector3d(const doubles& value);

/**
 * @brief Converts a vector of doubles into a 4D vector
 *
 * @param value Vector of doubles
 * @return Vector3d A 4D vector
 */
Vector4d doublesToVector4d(const doubles& value);

/**
 * @brief Converts a vector of doubles into a Quaternion
 *
 * @param values Vector of doubles
 * @return Quaternion A quaternion
 */
Quaterniond doublesToQuaterniond(const doubles& values);

/**
 * @brief Converts a vector of doubles into vector of 4D vectors
 *
 * @param values Vector of doubles
 * @return Quaternion A vector of 4D vectors
 */
Vector4ds doublesToVector4ds(const doubles& values);

/**
 * @brief Converts a vector of doubles into animation details
 *
 * @param value Vector of doubles
 * @return AnimationDetails The animation details
 */
AnimationDetails doublesToAnimationDetails(const doubles& values);

/**
 * @brief Returns a position and a rotation of a instance on a sphere using a
 * sphere-filling algorythm
 *
 * @param radius Radius of the sphere
 * @param occurrence Occurence of the instance
 * @param occurrences Total number of instances
 * @param rnd Randomized occurence of the instance (optional)
 * @param position Resulting position of the instance on the sphere
 * @param rotation Resulting orientation of the instance on the sphere
 * @param ratio Ratio of coverage of the sphere
 * @return Vector3d
 */
Vector3d sphereFilling(const double radius, const uint64_t occurrence,
                       const uint64_t occurrences, const uint64_t rnd,
                       Vector3d& position, Quaterniond& rotation,
                       const double ratio = 1.0);

/**
 * @brief Splits a string according to the delimiter
 *
 * @param s String to split
 * @param delimiter Delimiter
 * @return std::vector<std::string> Vector of strings
 */
std::vector<std::string> split(
    const std::string& s, const std::string& delimiter = CONTENTS_DELIMITER);

/**
 * @brief Combine a list of transformations
 *
 * @param transformations List of transformations
 * @return Transformation Result of the combination
 */
Transformation combineTransformations(const Transformations& transformations);

/**
 * @brief Safely converts an orientation vector into a quaternion
 *
 * @param v Orientation vector
 * @return Quaterniond Resulting quaternion
 */
Quaterniond safeQuatlookAt(const Vector3d& v);

/**
 * @brief Intersection between a ray and a box
 *
 * @param origin Origin of the ray
 * @param direction Direcion of the ray
 * @param box Box
 * @param t0 Initial t of the ray
 * @param t1 Final t of the ray
 * @param t Intersection value of t if an intersection if found
 * @return true The ray intersects with the box
 * @return false The ray does not intersect with the box
 */
bool rayBoxIntersection(const Vector3d& origin, const Vector3d& direction,
                        const Boxd& box, const double t0, const double t1,
                        double& t);

/**
 * @brief Get the Bezier Point from a curve defined by the provided control
 * points
 *
 * @param controlPoints Curve control points with radius
 * @param t The t in the function for a curve can be thought of as describing
 * how far B(t) is from first to last control point.
 * @return Vector3f
 */
Vector4f getBezierPoint(const Vector4fs& controlPoints, const double t);

// Volumes
double sphereVolume(const double radius);
double cylinderVolume(const double height, const double radius);
double coneVolume(const double height, const double r1, const double r2);
double capsuleVolume(const double height, const double radius);

Vector3f transformVector3f(const Vector3f& v, const Matrix4f& transformation);
Vector3ds getPointsInSphere(const size_t nbPoints, const double innerRadius);

double frac(const double x);
Vector3d frac(const Vector3d x);
double mix(const double x, const double y, const double a);
double hash(const double n);
double noise(const Vector3d& x);
Vector3d mod(const Vector3d& v, const int m);
double cells(const Vector3d& p, const double cellCount);
double worleyNoise(const Vector3d& p, const double cellCount);

size_t getMaterialIdFromOrientation(const Vector3d& orientation);

/**
 * @brief Return a random double between -0.5 and 0.5
 *
 * @return double A random double between -0.5 and 0.5
 */
double rnd1();

/**
 * @brief Return a predefined random double between -0.5 and 0.5
 *
 * @param index Index of the random double in a predefined array
 * @return double A random double between -0.5 and 0.5
 */
double rnd2(const uint64_t index);

/**
 * @brief Return a controlled random double between -0.5 and 0.5, currently
 * a sinusoidal function
 *
 * @param index Index of the random double in a sinusoidal function
 * @return double A random double between -0.5 and 0.5
 */
double rnd3(const uint64_t index);

/**
 * @brief Randomly alters a quaternion according to the specified parameters
 *
 * @param q Initial quaternion
 * @param seed Random seed
 * @param index Index of the quaternion (typically the index of the
 * corresponding element instance)
 * @param weight Weight of the alteration
 * @return Quaterniond Resulting modified quaternion
 */
Quaterniond weightedRandomRotation(const Quaterniond& q, const uint64_t seed,
                                   const uint64_t index, const double weight);

/**
 * @brief Generate a random quaternion
 *
 * @param seed Seed to apply to the randomness
 * @return Quaterniond Random quaternion
 */
Quaterniond randomQuaternion(const uint64_t seed);

} // namespace common
} // namespace bioexplorer
