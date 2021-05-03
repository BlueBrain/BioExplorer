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

#include "Shapes.h"

#include <brayns/common/Transformation.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

const std::vector<float> randoms = {
    0.28148369141,    0.796861024715, 0.074193743197,  0.482440306945,
    0.992773878589,   0.709310247315, 0.988484235866,  0.714714091734,
    0.643116781373,   0.718637647581, 0.926101047171,  0.846328419497,
    0.00297438943897, 0.137931361741, 0.17772706582,   0.444643858689,
    0.629288179636,   0.613382480923, 0.630336849193,  0.311776793613,
    0.08508451954,    0.30789230424,  0.0498092039943, 0.773779960365,
    0.768769637233,   0.882161981223, 0.976723516158,  0.449556805562,
    0.817669534955,   0.616539655821, 0.758216742242,  0.858237417116,
    0.979179183398,   0.65720513278,  0.386168029804,  0.0998493897615,
    0.962177647248,   0.108548816296, 0.996156105474,  0.941749314739,
    0.406174983692,   0.158989971035, 0.654907085688,  0.538001003242,
    0.332477591342,   0.978302973988, 0.98409103864,   0.241245008961,
    0.68183193795,    0.653235229058, 0.0606653606997, 0.0566309454523,
    0.919881491327,   0.905670025614, 0.637338702024,  0.121894161196,
    0.937476480417,   0.017741798193, 0.61697799368,   0.709261525057,
    0.859211525517,   0.96409034113,  0.0972400297964, 0.181073145261,
    0.284798532204,   0.413248667128, 0.332659388212,  0.340977212815,
    0.820090638467,   0.560592082547, 0.183689859617,  0.2575201395,
    0.289725466835,   0.522736633275, 0.882031679296,  0.654563598748,
    0.531309473163,   0.134963142807, 0.601297763714,  0.483506281956,
    0.283419807601,   0.454826306306, 0.508528602139,  0.897831546117,
    0.900287116387,   0.688215721818, 0.615842816633,  0.78273583615,
    0.927051829764,   0.425934500525, 0.741948788292,  0.0813684454157,
    0.998899378243,   0.551326196783, 0.0682702415237, 0.389893584905,
    0.15548746549,    0.468047910542, 0.948034950244,  0.202074251433,
    0.347536181502,   0.024377007386, 0.2214820153,    0.846643514875,
    0.391710310296,   0.692284401129, 0.244449478476,  0.0181219259474,
    0.336741055884,   0.70325501105,  0.968370058703,  0.892508506776,
    0.538387343968,   0.843838154621, 0.0790397063184, 0.103191163974,
    0.243711484807,   0.694622402023, 0.798540922368,  0.21746310996,
    0.870761691473,   0.368350833275, 0.228505271004,  0.3741636072,
    0.347291149036,   0.753449262487, 0.890757112194,  0.167150644248};

float rnd1()
{
    return float(rand() % 1000 - 500) / 1000.f;
}

float rnd2(const size_t index)
{
    return randoms[index % randoms.size()] - 0.5f;
}

float rnd3(const size_t index)
{
    return cos(index * M_PI / 180.f) + sin(index * M_PI / 45.f);
}

Quaterniond randomQuaternion(const size_t seed)
{
    double x, y, z, u, v, w, s;
    do
    {
        x = rnd2(seed);
        y = rnd2(seed + 1);
        z = x * x + y * y;
    } while (z > 1.0);
    do
    {
        u = rnd2(seed + 2);
        v = rnd2(seed + 3);
        w = u * u + v * v;
    } while (w > 1.0);
    s = sqrt((1.0 - z) / w);
    return Quaterniond(x, y, s * u, s * v);
}

Transformation getSphericalPosition(const Vector3f& position,
                                    const float radius, const size_t occurence,
                                    const size_t occurences,
                                    const RandomizationDetails& randInfo)
{
    size_t rnd = occurence;
    if (occurences != 0 && randInfo.seed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurences;

    const float offset = 2.f / occurences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    // Position randomizer
    float R = radius;
    if (randInfo.positionSeed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::radial)
        R += randInfo.positionStrength * rnd3(randInfo.positionSeed + rnd);

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + offset / 2.f;
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = rnd * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    Vector3f d{x, y, z};
    Vector3f pos;
    if (randInfo.randomizationType == PositionRandomizationType::radial)
        pos = (R + position.y) * d;
    else
        pos = position + R * d;

    // rotation randomizer
    if (randInfo.rotationSeed != 0)
        d = d + randInfo.rotationStrength *
                    Vector3f(rnd2(randInfo.rotationSeed + rnd * 2),
                             rnd2(randInfo.rotationSeed + rnd * 3),
                             rnd2(randInfo.rotationSeed + rnd * 5));

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(quatLookAt(normalize(d), UP_VECTOR));
    return transformation;
}

Transformation getFanPosition(const Vector3f& position, const float radius,
                              const size_t occurence, const size_t occurences,
                              const RandomizationDetails& randInfo)
{
    size_t rnd = occurence;
    if (occurences != 0 && randInfo.seed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurences;

    const float offset = 2.f / occurences;
    const float increment = 0.1f * M_PI * (3.f - sqrt(5.f));

    // Randomizer
    float R = radius;
    if (randInfo.seed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::radial)
        R *= 1.f + rnd1() / 30.f;

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + offset / 2.f;
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = rnd * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    const Vector3f d{x, y, z};

    Transformation transformation;
    transformation.setTranslation(position + R * d);
    transformation.setRotation(quatLookAt(d, UP_VECTOR));
    return transformation;
}

Transformation getPlanarPosition(const Vector3f& position, const float size,
                                 const RandomizationDetails& randInfo)
{
    float up = 0.f;
    if (randInfo.seed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::radial)
        up = rnd1() / 20.f;

    Transformation transformation;
    transformation.setTranslation(position +
                                  Vector3f(rnd1() * size, up, rnd1() * size));
    transformation.setRotation(quatLookAt({0.f, 1.f, 0.f}, UP_VECTOR));
    return transformation;
}

Transformation getCubicPosition(const Vector3f& position, const float size,
                                const RandomizationDetails& randInfo)
{
    Vector3f pos =
        position + Vector3f(rnd1() * size, rnd1() * size, rnd1() * size);
    Quaterniond dir = quatLookAt({rnd1(), rnd1(), rnd1()}, UP_VECTOR);

    if (randInfo.positionSeed != 0)
    {
        const Vector3f posOffset = randInfo.positionStrength *
                                   Vector3f(rnd2(randInfo.positionSeed),
                                            rnd2(randInfo.positionSeed + 1),
                                            rnd2(randInfo.positionSeed + 2));

        pos += posOffset;
    }

    if (randInfo.rotationSeed != 0)
        dir = dir + randomQuaternion(randInfo.rotationSeed);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(dir);
    return transformation;
}

float sinusoide(const float x, const float z)
{
    return 0.2f * cos(x) * sin(z) + 0.05f * cos(x * 2.3f) * sin(z * 4.6f);
}

Transformation getSinosoidalPosition(const Vector3f& position, const float size,
                                     const float amplitude,
                                     const size_t occurence,
                                     const RandomizationDetails& randInfo)
{
    const float step = 0.01f;
    const float angle = 0.01f;
    float upOffset = 0.f;
    if (randInfo.positionSeed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::radial)
        upOffset = randInfo.positionStrength *
                   rnd3((randInfo.positionSeed + occurence) * 10);

    const float x = rnd1() * size;
    const float z = rnd1() * size;
    const float y = upOffset + amplitude * sinusoide(x * angle, z * angle);

    Vector3f pos = Vector3f(x, y, z);

    const Vector3f v1 =
        Vector3f(x + step,
                 upOffset +
                     amplitude * sinusoide((x + step) * angle, z * angle),
                 z) -
        pos;
    const Vector3f v2 =
        Vector3f(x,
                 upOffset +
                     amplitude * sinusoide(x * angle, (z + step) * angle),
                 z + step) -
        pos;

    pos += position;

    Vector3f d = cross(normalize(v1), normalize(v2));
    if (randInfo.rotationSeed != 0)
        d = d + randInfo.rotationStrength *
                    Vector3f(rnd2(randInfo.rotationSeed + occurence + 1),
                             rnd2(randInfo.rotationSeed + occurence + 2),
                             rnd2(randInfo.rotationSeed + occurence + 3));

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(quatLookAt(normalize(d), UP_VECTOR));
    return transformation;
}

Transformation getBezierPosition(const Vector3fs& points, const float scale,
                                 const float t)
{
    Vector3fs bezierPoints = points;
    for (auto& bezierPoint : bezierPoints)
        bezierPoint *= scale;

    size_t i = bezierPoints.size() - 1;
    while (i > 0)
    {
        for (size_t k = 0; k < i; ++k)
            bezierPoints[k] =
                bezierPoints[k] + t * (bezierPoints[k + 1] - bezierPoints[k]);
        --i;
    }

    Transformation transformation;
    transformation.setTranslation(bezierPoints[0]);
    transformation.setRotation(quatLookAt(
        normalize(cross({0.f, 0.f, 1.f}, bezierPoints[1] - bezierPoints[0])),
        UP_VECTOR));
    return transformation;
}

Transformation getSphericalToPlanarPosition(
    const Vector3f& center, const float radius, const size_t occurence,
    const size_t occurences, const RandomizationDetails& randInfo,
    const float morphingStep)
{
    size_t rnd = occurence;
    if (occurences != 0 && randInfo.seed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::circular)
        rnd = rand() % occurences;

    const float offset = 2.f / occurences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    // Position randomizer
    float R = radius;
    if (randInfo.positionSeed != 0 &&
        randInfo.randomizationType == PositionRandomizationType::radial)
        R += randInfo.positionStrength * rnd3(randInfo.positionSeed + rnd);

    // Sphere filling
    const float y = ((rnd * offset) - 1.f) + (offset / 2.f);
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = rnd * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;

    Vector3f startPos;
    Vector3f startDir = Vector3f(x, y, z);
    if (randInfo.randomizationType == PositionRandomizationType::radial)
        startPos = (R + center.y) * startDir;
    else
        startPos = center + R * startDir;

    Vector3f endPos = startPos;

    // rotation randomizer
    if (randInfo.rotationSeed != 0)
        startDir =
            startDir + randInfo.rotationStrength *
                           Vector3f(rnd2(randInfo.rotationSeed + rnd * 2),
                                    rnd2(randInfo.rotationSeed + rnd * 3),
                                    rnd2(randInfo.rotationSeed + rnd * 5));

    R = radius;
    const float endRadius = R * 1.75f;

    endPos.y = -R;
    endPos = endPos + (1.f - (startPos.y + R) / endRadius) *
                          Vector3f(endRadius, 0.f, endRadius) *
                          normalize(Vector3f(startDir.x, 0.f, startDir.z));

    const Vector3f endDir{0.f, 1.f, 0.f};
    const Vector3f d = startDir + morphingStep * (endDir - startDir);

    Transformation transformation;
    transformation.setTranslation(endPos * morphingStep +
                                  startPos * (1.f - morphingStep));
    transformation.setRotation(quatLookAt(normalize(d), UP_VECTOR));
    return transformation;
}
} // namespace common
} // namespace bioexplorer
