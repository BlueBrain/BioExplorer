/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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
#include <brayns/engineapi/Model.h>

#include <glm/gtx/matrix_decompose.hpp>

namespace bioexplorer
{
namespace common
{
using namespace brayns;

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

Quaterniond safeQuatlookAt(const Vector3d& v)
{
    const Vector3d vector = normalize(v);
    auto upVector = UP_VECTOR;
    if (abs(dot(vector, upVector)) > 0.999)
        // Gimble lock
        upVector = Vector3d(0.0, 0.0, 1.0);
    return quatLookAtRH(vector, upVector);
}

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

bool isClipped(const Vector3d& position, const Vector4ds& clippingPlanes)
{
    if (clippingPlanes.empty())
        return false;

    bool visible = true;
    for (auto plane : clippingPlanes)
    {
        const Vector3d normal = normalize(Vector3d(plane.x, plane.y, plane.z));
        const double d = plane.w;
        const double distance = dot(normal, position) - d;
        visible &= (distance < 0.0);
    }
    return !visible;
}

void setDefaultTransferFunction(Model& model, const Vector2d range)
{
    TransferFunction& tf = model.getTransferFunction();
    tf.setControlPoints({{0.0, 0.5}, {1.0, 0.5}});
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
    tf.setValuesRange(range);
}

Vector4ds getClippingPlanes(const Scene& scene)
{
    const auto& clippingPlanes = scene.getClipPlanes();
    Vector4ds clipPlanes;
    for (const auto cp : clippingPlanes)
    {
        const auto& p = cp->getPlane();
        Vector4d plane{p[0], p[1], p[2], p[3]};
        clipPlanes.push_back(plane);
    }
    return clipPlanes;
}

Vector2d doublesToVector2d(const doubles& value)
{
    if (value.empty())
        return Vector2d();
    if (value.size() != 2)
        PLUGIN_THROW("Invalid number of doubles (2 expected)");
    return Vector2d(value[0], value[1]);
}

Vector3d doublesToVector3d(const doubles& value)
{
    if (value.empty())
        return Vector3d();
    if (value.size() != 3)
        PLUGIN_THROW("Invalid number of doubles (3 expected)");
    return Vector3d(value[0], value[1], value[2]);
}

Vector4d doublesToVector4d(const doubles& value)
{
    if (value.empty())
        return Vector4d();
    if (value.size() != 4)
        PLUGIN_THROW("Invalid number of doubles (4 expected)");
    return Vector4d(value[0], value[1], value[2], value[3]);
}

Quaterniond doublesToQuaterniond(const doubles& values)
{
    if (values.empty())
        return Quaterniond();
    if (values.size() != 4)
        PLUGIN_THROW("Invalid number of doubles (4 expected)");
    return Quaterniond(values[0], values[1], values[2], values[3]);
}

Vector4ds doublesToVector4ds(const doubles& values)
{
    if (values.empty())
        return Vector4ds();
    if (values.size() % 4 != 0)
        PLUGIN_THROW("Clipping planes must be defined by 4 double values");

    Vector4ds clippingPlanes;
    for (size_t i = 0; i < values.size(); i += 4)
        clippingPlanes.push_back(
            {values[i], values[i + 1], values[i + 2], values[i + 3]});
    return clippingPlanes;
}

MolecularSystemAnimationDetails doublesToMolecularSystemAnimationDetails(
    const doubles& values)
{
    MolecularSystemAnimationDetails details;
    details.seed = (values.size() > 0 ? values[0] : 0);
    details.positionSeed = (values.size() > 1 ? values[1] : 0);
    details.positionStrength = (values.size() > 2 ? values[2] : 0.0);
    details.rotationSeed = (values.size() > 3 ? values[3] : 0);
    details.rotationStrength = (values.size() > 4 ? values[4] : 0.0);
    details.morphingStep = (values.size() > 5 ? values[5] : 0.0);
    return details;
}

CellAnimationDetails doublesToCellAnimationDetails(const doubles& values)
{
    CellAnimationDetails details;
    details.seed = (values.size() > 0 ? values[0] : 0);
    details.offset = (values.size() > 1 ? values[1] : 0);
    details.amplitude = (values.size() > 2 ? values[2] : 1.0);
    details.frequency = (values.size() > 3 ? values[3] : 1.0);
    return details;
}

std::vector<std::string> split(const std::string& s,
                               const std::string& delimiter)
{
    std::vector<std::string> values;
    if (s.empty())
        return values;

    std::string str = s;
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos)
    {
        token = str.substr(0, pos);
        values.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    values.push_back(str);
    return values;
}

Transformation combineTransformations(const Transformations& transformations)
{
    glm::mat4 finalMatrix;
    for (const auto& transformation : transformations)
    {
        const glm::mat4 matrix = transformation.toMatrix();
        finalMatrix *= matrix;
    }

    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translation;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(finalMatrix, scale, rotation, translation, skew,
                   perspective);

    Transformation transformation;
    transformation.setTranslation(translation);
    transformation.setRotation(rotation);
    // transformation.setScale(scale);
    return transformation;
}

Vector3d sphereFilling(const double radius, const uint64_t occurrence,
                       const uint64_t occurrences, const uint64_t rnd,
                       Vector3d& position, Quaterniond& rotation,
                       const double ratio)
{
    const double off = 2.0 / occurrences;
    const double increment = ratio * M_PI * (3.0 - sqrt(5.0));
    const double y = ((occurrence * off) - 1.0) + off / 2.0;
    const double r = sqrt(1.0 - pow(y, 2.0));
    const double phi = rnd * increment;
    const double x = cos(phi) * r;
    const double z = sin(phi) * r;

    const Vector3d normal = normalize(Vector3d(x, y, z));
    position = normal * radius;
    rotation = safeQuatlookAt(normal);

    return normal;
}

bool rayBoxIntersection(const Vector3d& origin, const Vector3d& direction,
                        const Boxd& box, const double t0, const double t1,
                        double& t)
{
    const Vector3d bounds[2]{box.getMin(), box.getMax()};
    const Vector3d invDir = 1.0 / direction;
    const Vector3ui sign{invDir.x < 0.0, invDir.y < 0.0, invDir.z < 0.0};

    double tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[sign.x].x - origin.x) * invDir.x;
    tmax = (bounds[1 - sign.x].x - origin.x) * invDir.x;
    tymin = (bounds[sign.y].y - origin.y) * invDir.y;
    tymax = (bounds[1 - sign.y].y - origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[sign.z].z - origin.z) * invDir.z;
    tzmax = (bounds[1 - sign.z].z - origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    t = std::min(tmin, tmax);

    return (tmin < t1 && tmax > t0);
}

Vector4f getBezierPoint(const Vector4fs& controlPoints, const double t)
{
    if (t < 0.0 || t > 1.0)
        PLUGIN_THROW("Invalid value with t=" + std::to_string(t) +
                     ". Must be between 0 and 1");
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

double sphereVolume(const double radius)
{
    return 4.0 * M_PI * pow(radius, 3) / 3.0;
}

double cylinderVolume(const double height, const double radius)
{
    return height * M_PI * radius * radius;
}

double coneVolume(const double height, const double r1, const double r2)
{
    return M_PI * (r1 * r1 + r1 * r2 + r2 * r2) * height / 3.0;
}

double capsuleVolume(const double height, const double radius)
{
    return sphereVolume(radius) + cylinderVolume(height, radius);
}

Vector3f transformVector3f(const Vector3f& v, const Matrix4f& transformation)
{
    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translation;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(transformation, scale, rotation, translation, skew,
                   perspective);
    return translation + rotation * v;
}

Vector3ds getPointsInSphere(const size_t nbPoints, const double innerRadius)
{
    const double radius =
        innerRadius + (rand() % 1000 / 1000.0) * (1.0 - innerRadius);
    double phi = M_PI * ((rand() % 2000 - 1000) / 1000.0);
    double theta = M_PI * ((rand() % 2000 - 1000) / 1000.0);
    Vector3ds points;
    for (size_t i = 0; i < nbPoints; ++i)
    {
        Vector3d point = {radius * sin(phi) * cos(theta),
                          radius * sin(phi) * sin(theta), radius * cos(phi)};
        points.push_back(point);
        phi += ((rand() % 1000) / 5000.0);
        theta += ((rand() % 1000) / 5000.0);
    }
    return points;
}

double mix(const double x, const double y, const double a)
{
    return x * (1 - a) + y * a;
}

double frac(const double x)
{
    return x - floor(x);
}

Vector3d frac(const Vector3d v)
{
    return Vector3d(v.x - floor(v.x), v.y - floor(v.y), v.z - floor(v.z));
}

double hash(double n)
{
    return frac(sin(n + 1.951) * 43758.5453);
}

double noise(const Vector3d& x)
{
    // hash based 3d value noise
    Vector3d p = floor(x);
    Vector3d f = frac(x);

    f = f * f * (Vector3d(3.0, 3.0, 3.0) - Vector3d(2.0, 2.0, 2.0) * f);
    double n = p.x + p.y * 57 + 113 * p.z;
    return mix(mix(mix(hash(n + 0), hash(n + 1), f.x),
                   mix(hash(n + 57), hash(n + 58), f.x), f.y),
               mix(mix(hash(n + 113), hash(n + 114), f.x),
                   mix(hash(n + 170), hash(n + 171), f.x), f.y),
               f.z);
}

Vector3d mod(const Vector3d& v, const int m)
{
    return Vector3d(v.x - m * floor(v.x / m), v.y - m * floor(v.y / m),
                    v.z - m * floor(v.z / m));
}

double cells(const Vector3d& p, const double cellCount)
{
    const Vector3d pCell = p * cellCount;
    double d = 1.0e10;
    for (int64_t xo = -1; xo <= 1; xo++)
    {
        for (int64_t yo = -1; yo <= 1; yo++)
        {
            for (int64_t zo = -1; zo <= 1; zo++)
            {
                Vector3d tp = floor(pCell) + Vector3d(xo, yo, zo);
                tp = pCell - tp - noise(mod(tp, cellCount / 1));
                d = std::min(d, dot(tp, tp));
            }
        }
    }
    d = std::min(d, 1.0);
    d = std::max(d, 0.0);
    return d;
}

double worleyNoise(const Vector3d& p, double cellCount)
{
    return cells(p, cellCount);
}

size_t getMaterialIdFromOrientation(const Vector3d& orientation)
{
    const Vector3d n = normalize(orientation);
    const Vector3ui rgb = 255.0 * (0.5 + 0.5 * n);
    return ((rgb.x & 0x0ff) << 16) | ((rgb.y & 0x0ff) << 8) | (rgb.z & 0x0ff);
}

double rnd1()
{
    return static_cast<double>(rand() % 1000 - 500) / 1000.0;
}

double rnd2(const uint64_t index)
{
    return randoms[index % randoms.size()] - 0.5;
}

double rnd3(const uint64_t index)
{
    return cos(index * M_PI / 180.0) + sin(index * M_PI / 45.0) +
           cos(index * M_PI / 72.0);
}

Quaterniond weightedRandomRotation(const Quaterniond& q, const uint64_t seed,
                                   const uint64_t index, const double weight)
{
    const Quaterniond qPitch =
        angleAxis(weight * rnd2(seed + index * 2), Vector3d(1.0, 0.0, 0.0));
    const Quaterniond qYaw =
        angleAxis(weight * rnd2(seed + index * 3), Vector3d(0.0, 1.0, 0.0));
    const Quaterniond qRoll =
        angleAxis(weight * rnd2(seed + index * 5), Vector3d(0.0, 0.0, 1.0));
    return q * qPitch * qYaw * qRoll;
}

Quaterniond randomQuaternion(const uint64_t seed)
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

bool andCheck(const uint32_t value, const uint32_t test)
{
    return (value & test) == test;
}

std::string boolAsString(const bool value)
{
    return (value ? "Yes" : "No");
}

} // namespace common
} // namespace bioexplorer
