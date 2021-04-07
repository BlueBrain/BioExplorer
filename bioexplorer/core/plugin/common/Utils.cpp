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

#include "Utils.h"

#include <plugin/common/Logs.h>

#include <brayns/common/scene/ClipPlane.h>
#include <brayns/common/transferFunction/TransferFunction.h>

namespace
{
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
} // namespace

namespace bioexplorer
{
std::string& ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::ptr_fun<int, int>(std::isgraph)));
    return s;
}

std::string& rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::ptr_fun<int, int>(std::isgraph))
                .base(),
            s.end());
    return s;
}

std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes)
{
    if (clippingPlanes.empty())
        return false;

    bool visible = true;
    for (auto plane : clippingPlanes)
    {
        const Vector3f normal = {plane.x, plane.y, plane.z};
        const float d = plane.w;
        const float distance = dot(normal, position) + d;
        visible &= (distance > 0.f);
    }
    return !visible;
}

void getSphericalPosition(
    const size_t rnd, const float assemblyRadius, const size_t occurence,
    const size_t occurences, const PositionRandomizationType randomizationType,
    const size_t randomPositionSeed, const float randomPositionStength,
    const size_t randomOrientationSeed, const float randomOrientationStength,
    const Vector3f& position, Vector3f& pos, Quaterniond& dir)
{
    const float offset = 2.f / occurences;
    const float increment = M_PI * (3.f - sqrt(5.f));
    const size_t index = (occurence + rnd) % occurences;

    // Position randomizer
    float radius = assemblyRadius;
    if (randomPositionSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        radius *=
            1.f + randomPositionStength * rnd3(randomPositionSeed + index);

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + (offset / 2.f);
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = index * increment;
    const float x = cos(phi) * r + assemblyRadius * 0.001f * rnd1();
    const float z = sin(phi) * r + assemblyRadius * 0.001f * rnd1();
    Vector3f d{x, y, z};
    if (randomizationType == PositionRandomizationType::radial)
        pos = (radius + position.y) * d;
    else
        pos = position + radius * d;

    // Orientation randomizer
    if (randomOrientationSeed != 0)
        d = d + randomOrientationStength *
                    Vector3f(rnd2(randomOrientationSeed + index * 2),
                             rnd2(randomOrientationSeed + index * 3),
                             rnd2(randomOrientationSeed + index * 5));
    dir = glm::quatLookAt(normalize(d), UP_VECTOR);
}

void getFanPosition(const size_t rnd, const float assemblyRadius,
                    const PositionRandomizationType randomizationType,
                    const size_t randomSeed, const size_t occurence,
                    const size_t occurences, const Vector3f& position,
                    Vector3f& pos, Quaterniond& dir)
{
    const float offset = 2.f / occurences;
    const float increment = 0.1f * M_PI * (3.f - sqrt(5.f));

    // Randomizer
    float radius = assemblyRadius;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        radius *= 1.f + (float(rand() % 1000 - 500) / 30000.f);

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + (offset / 2.f);
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = ((occurence + rnd) % occurences) * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    const Vector3f d{x, y, z};
    pos = position + radius * d;
    dir = glm::quatLookAt(d, UP_VECTOR);
}

void getPlanarPosition(const float assemblyRadius,
                       const PositionRandomizationType randomizationType,
                       const size_t randomSeed, const Vector3f& position,
                       Vector3f& pos, Quaterniond& dir)
{
    float up = 0.f;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        up = (float(rand() % 1000 - 500) / 20000.f);

    pos = position +
          Vector3f(float(rand() % 1000 - 500) / 1000.f * assemblyRadius, up,
                   float(rand() % 1000 - 500) / 1000.f * assemblyRadius);
    dir = glm::quatLookAt({0.f, 1.f, 0.f}, UP_VECTOR);
}

void getCubicPosition(const float size, const Vector3f& position,
                      const size_t randomPositionSeed,
                      const float randomPositionStength,
                      const size_t randomOrientationSeed,
                      const float randomOrientationStength, Vector3f& pos,
                      Quaterniond& dir)
{
    pos = position + Vector3f(rnd1() * size, rnd1() * size, rnd1() * size);
    dir = glm::quatLookAt({rnd1(), rnd1(), rnd1()}, UP_VECTOR);
    if (randomOrientationSeed != 0)
        dir = dir + randomQuaternion(randomOrientationSeed);

    if (randomPositionSeed != 0)
    {
        const Vector3f posOffset =
            randomPositionStength * Vector3f(rnd2(randomOrientationSeed + 3),
                                             rnd2(randomOrientationSeed + 4),
                                             rnd2(randomOrientationSeed + 5));

        pos += posOffset;
    }
}

float sinusoide(const float x, const float z)
{
    return 0.2f * cos(x) * sin(z) + 0.05f * cos(x * 2.3f) * sin(z * 4.6f);
}

void getSinosoidalPosition(
    const float size, const float amplitude, const size_t occurence,
    const PositionRandomizationType randomizationType,
    const size_t randomPositionSeed, const float randomPositionStrength,
    const size_t randomOrientationSeed, const float randomOrientationStrength,
    const Vector3f& position, Vector3f& pos, Quaterniond& dir)
{
    const float step = 0.01f;
    const float angle = 0.01f;
    float up = 1.f;
    if (randomPositionSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        up = 1.f + randomPositionStrength *
                       rnd3((randomOrientationSeed + occurence) * 10);

    const float x = rnd1() * size;
    const float z = rnd1() * size;
    const float y = amplitude * up * sinusoide(x * angle, z * angle);

    pos = Vector3f(x, y, z);

    const Vector3f v1 =
        Vector3f(x + step,
                 amplitude * up * sinusoide((x + step) * angle, z * angle), z) -
        pos;
    const Vector3f v2 =
        Vector3f(x, amplitude * up * sinusoide(x * angle, (z + step) * angle),
                 z + step) -
        pos;

    pos += position;

    Vector3f d = cross(normalize(v1), normalize(v2));
    if (randomOrientationSeed != 0)
        d = d + randomOrientationStrength *
                    Vector3f(rnd2(randomOrientationSeed + occurence + 1),
                             rnd2(randomOrientationSeed + occurence + 2),
                             rnd2(randomOrientationSeed + occurence + 3));
    dir = glm::quatLookAt(normalize(d), UP_VECTOR);
}

void getBezierPosition(const Vector3fs& points, const float assemblyRadius,
                       const float t, Vector3f& pos, Quaterniond& dir)
{
    Vector3fs bezierPoints = points;
    for (auto& bezierPoint : bezierPoints)
        bezierPoint *= assemblyRadius;

    size_t i = bezierPoints.size() - 1;
    while (i > 0)
    {
        for (size_t k = 0; k < i; ++k)
            bezierPoints[k] =
                bezierPoints[k] + t * (bezierPoints[k + 1] - bezierPoints[k]);
        --i;
    }
    dir = glm::quatLookAt(normalize(cross({0, 0, 1},
                                          bezierPoints[1] - bezierPoints[0])),
                          UP_VECTOR);
    pos = bezierPoints[0];
}

void getSphericalToPlanarPosition(
    const size_t rnd, const float assemblyRadius, const size_t occurence,
    const size_t occurences, const PositionRandomizationType randomizationType,
    const size_t randomPositionSeed, const float randomPositionStrengh,
    const size_t randomOrientationSeed, const float randomOrientationStrengh,
    const Vector3f& position, const float morphingStep, Vector3f& pos,
    Quaterniond& dir)
{
    Vector3f startPos;
    Quaterniond startDir;
    getSphericalPosition(rnd, assemblyRadius, occurence, occurences,
                         randomizationType, randomPositionSeed,
                         randomPositionStrengh, randomOrientationSeed,
                         randomOrientationStrengh, {0, 0, 0}, startPos,
                         startDir);
    Vector3f endPos;

    endPos = startPos;
    endPos.z = -assemblyRadius;
    endPos = endPos +
             (1.f - (startPos.z + assemblyRadius) / (assemblyRadius * 2.f)) *
                 Vector3f(assemblyRadius * 2.f, assemblyRadius * 2.f, 0.f) *
                 normalize(Vector3f(startDir.x, startDir.y, 0.f));

    pos = (endPos * morphingStep + startPos * (1.f - morphingStep));
    dir = startDir;
}

void setTransferFunction(brayns::TransferFunction& tf)
{
    tf.setControlPoints({{0.0, 0.0}, {0.1, 1.0}, {1.0, 1.0}});
    // curl https://api.colormaps.io/colormap/unipolar
    tf.setColorMap(
        {"unipolar",
         {{0.0, 0.0, 0.0},
          {0.00392156862745098, 0.00392156862745098, 0.12941176470588237},
          {0.00784313725490196, 0.00784313725490196, 0.25882352941176473},
          {0.011764705882352941, 0.011764705882352941, 0.39215686274509803},
          {0.01568627450980392, 0.01568627450980392, 0.5215686274509804},
          {0.0196078431372549, 0.0196078431372549, 0.6549019607843137},
          {0.03529411764705882, 0.0784313725490196, 0.6862745098039216},
          {0.047058823529411764, 0.13333333333333333, 0.7215686274509804},
          {0.058823529411764705, 0.18823529411764706, 0.7568627450980392},
          {0.07450980392156863, 0.24705882352941178, 0.788235294117647},
          {0.08627450980392157, 0.30196078431372547, 0.8235294117647058},
          {0.09803921568627451, 0.3607843137254902, 0.8588235294117647},
          {0.11372549019607843, 0.41568627450980394, 0.8901960784313725},
          {0.12549019607843137, 0.47058823529411764, 0.9254901960784314},
          {0.13725490196078433, 0.5294117647058824, 0.9568627450980393},
          {0.2196078431372549, 0.4666666666666667, 0.8745098039215686},
          {0.30196078431372547, 0.403921568627451, 0.796078431372549},
          {0.3843137254901961, 0.3411764705882353, 0.7137254901960784},
          {0.4823529411764706, 0.28627450980392155, 0.596078431372549},
          {0.5764705882352941, 0.22745098039215686, 0.47843137254901963},
          {0.6705882352941176, 0.16862745098039217, 0.36470588235294116},
          {0.7686274509803922, 0.11372549019607843, 0.24705882352941178},
          {0.8627450980392157, 0.054901960784313725, 0.13333333333333333},
          {0.9568627450980393, 0.0, 0.01568627450980392},
          {0.9568627450980393, 0.0196078431372549, 0.01568627450980392},
          {0.9529411764705882, 0.043137254901960784, 0.01568627450980392},
          {0.9490196078431372, 0.06666666666666667, 0.01568627450980392},
          {0.9450980392156862, 0.08627450980392157, 0.01568627450980392},
          {0.9411764705882353, 0.10980392156862745, 0.01568627450980392},
          {0.9372549019607843, 0.13333333333333333, 0.011764705882352941},
          {0.9333333333333333, 0.1568627450980392, 0.011764705882352941},
          {0.9333333333333333, 0.17647058823529413, 0.011764705882352941},
          {0.9294117647058824, 0.2, 0.011764705882352941},
          {0.9254901960784314, 0.2235294117647059, 0.011764705882352941},
          {0.9215686274509803, 0.24705882352941178, 0.011764705882352941},
          {0.9176470588235294, 0.26666666666666666, 0.00784313725490196},
          {0.9137254901960784, 0.2901960784313726, 0.00784313725490196},
          {0.9098039215686274, 0.3137254901960784, 0.00784313725490196},
          {0.9098039215686274, 0.33725490196078434, 0.00784313725490196},
          {0.9058823529411765, 0.3568627450980392, 0.00784313725490196},
          {0.9019607843137255, 0.3803921568627451, 0.00784313725490196},
          {0.8980392156862745, 0.403921568627451, 0.00392156862745098},
          {0.8941176470588236, 0.4235294117647059, 0.00392156862745098},
          {0.8901960784313725, 0.4470588235294118, 0.00392156862745098},
          {0.8862745098039215, 0.47058823529411764, 0.00392156862745098},
          {0.8823529411764706, 0.49411764705882355, 0.00392156862745098},
          {0.8823529411764706, 0.5137254901960784, 0.00392156862745098},
          {0.8784313725490196, 0.5372549019607843, 0.0},
          {0.8745098039215686, 0.5607843137254902, 0.0},
          {0.8705882352941177, 0.5843137254901961, 0.0},
          {0.8666666666666667, 0.6039215686274509, 0.0},
          {0.8627450980392157, 0.6274509803921569, 0.0},
          {0.8588235294117647, 0.6509803921568628, 0.0},
          {0.8588235294117647, 0.6745098039215687, 0.0},
          {0.8588235294117647, 0.6823529411764706, 0.01568627450980392},
          {0.8627450980392157, 0.6901960784313725, 0.03529411764705882},
          {0.8666666666666667, 0.7019607843137254, 0.050980392156862744},
          {0.8705882352941177, 0.7098039215686275, 0.07058823529411765},
          {0.8705882352941177, 0.7176470588235294, 0.08627450980392157},
          {0.8745098039215686, 0.7294117647058823, 0.10588235294117647},
          {0.8784313725490196, 0.7372549019607844, 0.12549019607843137},
          {0.8823529411764706, 0.7450980392156863, 0.1411764705882353},
          {0.8823529411764706, 0.7568627450980392, 0.1607843137254902},
          {0.8862745098039215, 0.7647058823529411, 0.17647058823529413},
          {0.8901960784313725, 0.7764705882352941, 0.19607843137254902},
          {0.8941176470588236, 0.7843137254901961, 0.21568627450980393},
          {0.8980392156862745, 0.792156862745098, 0.23137254901960785},
          {0.8980392156862745, 0.803921568627451, 0.25098039215686274},
          {0.9019607843137255, 0.8117647058823529, 0.26666666666666666},
          {0.9058823529411765, 0.8196078431372549, 0.28627450980392155},
          {0.9098039215686274, 0.8313725490196079, 0.3058823529411765},
          {0.9098039215686274, 0.8392156862745098, 0.3215686274509804},
          {0.9137254901960784, 0.8509803921568627, 0.3411764705882353},
          {0.9176470588235294, 0.8588235294117647, 0.3568627450980392},
          {0.9215686274509803, 0.8666666666666667, 0.3764705882352941},
          {0.9215686274509803, 0.8784313725490196, 0.396078431372549},
          {0.9254901960784314, 0.8862745098039215, 0.4117647058823529},
          {0.9294117647058824, 0.8941176470588236, 0.43137254901960786},
          {0.9333333333333333, 0.9058823529411765, 0.4470588235294118},
          {0.9372549019607843, 0.9137254901960784, 0.4666666666666667},
          {0.9372549019607843, 0.9254901960784314, 0.48627450980392156},
          {0.9411764705882353, 0.9333333333333333, 0.5019607843137255},
          {0.9450980392156862, 0.9411764705882353, 0.5215686274509804},
          {0.9490196078431372, 0.9529411764705882, 0.5372549019607843},
          {0.9490196078431372, 0.9607843137254902, 0.5568627450980392},
          {0.9529411764705882, 0.9686274509803922, 0.5764705882352941},
          {0.9568627450980393, 0.9803921568627451, 0.592156862745098},
          {0.9607843137254902, 0.9882352941176471, 0.611764705882353},
          {0.9647058823529412, 1.0, 0.6274509803921569},
          {0.9647058823529412, 1.0, 0.6392156862745098},
          {0.9647058823529412, 1.0, 0.6470588235294118},
          {0.9647058823529412, 1.0, 0.6588235294117647},
          {0.9647058823529412, 1.0, 0.6666666666666666},
          {0.9686274509803922, 1.0, 0.6745098039215687},
          {0.9686274509803922, 1.0, 0.6862745098039216},
          {0.9686274509803922, 1.0, 0.6941176470588235},
          {0.9686274509803922, 1.0, 0.7019607843137254},
          {0.9725490196078431, 1.0, 0.7137254901960784},
          {0.9725490196078431, 1.0, 0.7215686274509804},
          {0.9725490196078431, 1.0, 0.7294117647058823},
          {0.9725490196078431, 1.0, 0.7411764705882353},
          {0.9725490196078431, 1.0, 0.7490196078431373},
          {0.9764705882352941, 1.0, 0.7568627450980392},
          {0.9764705882352941, 1.0, 0.7686274509803922},
          {0.9764705882352941, 1.0, 0.7764705882352941},
          {0.9764705882352941, 1.0, 0.7843137254901961},
          {0.9803921568627451, 1.0, 0.796078431372549},
          {0.9803921568627451, 1.0, 0.803921568627451},
          {0.9803921568627451, 1.0, 0.8117647058823529},
          {0.9803921568627451, 1.0, 0.8235294117647058},
          {0.9803921568627451, 1.0, 0.8313725490196079},
          {0.984313725490196, 1.0, 0.8431372549019608},
          {0.984313725490196, 1.0, 0.8509803921568627},
          {0.984313725490196, 1.0, 0.8588235294117647},
          {0.984313725490196, 1.0, 0.8705882352941177},
          {0.9882352941176471, 1.0, 0.8784313725490196},
          {0.9882352941176471, 1.0, 0.8862745098039215},
          {0.9882352941176471, 1.0, 0.8980392156862745},
          {0.9882352941176471, 1.0, 0.9058823529411765},
          {0.9882352941176471, 1.0, 0.9137254901960784},
          {0.9921568627450981, 1.0, 0.9254901960784314},
          {0.9921568627450981, 1.0, 0.9333333333333333},
          {0.9921568627450981, 1.0, 0.9411764705882353},
          {0.9921568627450981, 1.0, 0.9529411764705882},
          {0.996078431372549, 1.0, 0.9607843137254902},
          {0.996078431372549, 1.0, 0.9686274509803922},
          {0.996078431372549, 1.0, 0.9803921568627451},
          {1.0, 1.0, 1.0}}});
    tf.setValuesRange({0.0, 1.0});
}

Vector4fs getClippingPlanes(const Scene& scene)
{
    const auto& clippingPlanes = scene.getClipPlanes();
    Vector4fs clipPlanes;
    for (const auto cp : clippingPlanes)
    {
        const auto& p = cp->getPlane();
        Vector4f plane{p[0], p[1], p[2], p[3]};
        clipPlanes.push_back(plane);
    }
    return clipPlanes;
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

} // namespace bioexplorer
