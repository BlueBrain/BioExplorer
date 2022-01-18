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

#include "Shape.h"

#include <plugin/common/Logs.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

const std::vector<double> randoms = {
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

Shape::Shape(const Vector4ds& clippingPlanes)
    : _clippingPlanes(clippingPlanes)
    , _surface(0.0)
{
}

Shape::~Shape() {}

double Shape::rnd1()
{
    return static_cast<double>(rand() % 1000 - 500) / 1000.0;
}

double Shape::rnd2(const uint64_t index)
{
    return randoms[index % randoms.size()] - 0.5;
}

double Shape::rnd3(const uint64_t index)
{
    return cos(index * M_PI / 180.0) + sin(index * M_PI / 45.0) +
           cos(index * M_PI / 72.0);
}

Quaterniond Shape::weightedRandomRotation(const Quaterniond& q,
                                          const uint64_t seed,
                                          const uint64_t index,
                                          const double weight)
{
    const Quaterniond qPitch =
        angleAxis(weight * rnd2(seed + index * 2), Vector3d(1.0, 0.0, 0.0));
    const Quaterniond qYaw =
        angleAxis(weight * rnd2(seed + index * 3), Vector3d(0.0, 1.0, 0.0));
    const Quaterniond qRoll =
        angleAxis(weight * rnd2(seed + index * 5), Vector3d(0.0, 0.0, 1.0));
    return q * qPitch * qYaw * qRoll;
}

Quaterniond Shape::randomQuaternion(const uint64_t seed) const
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

} // namespace common
} // namespace bioexplorer
