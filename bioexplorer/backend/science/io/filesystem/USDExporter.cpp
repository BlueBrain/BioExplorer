/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "USDExporter.h"

#include <science/common/Logs.h>

#ifdef USE_PIXAR
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#endif // USE_PIXAR

namespace bioexplorer
{
namespace io
{
namespace filesystem
{
using namespace core;
using namespace morphology;
using namespace details;

#ifdef USE_PIXAR
PXR_NAMESPACE_USING_DIRECTIVE

// Create shared material

UsdShadeMaterial _createSharedMaterial(const UsdStageRefPtr& stage, const std::string& materialName,
                                       const GfVec3f& color)
{
    // Define the material
    UsdShadeMaterial material = UsdShadeMaterial::Define(stage, SdfPath("/Materials/material_" + materialName));

    // Create the shader (UsdPreviewSurface for basic shading)
    UsdShadeShader shader = UsdShadeShader::Define(stage, material.GetPath().AppendChild(TfToken("PreviewSurface")));
    shader.CreateIdAttr().Set(TfToken("UsdPreviewSurface"));

    // Set the shader diffuse color
    shader.CreateInput(TfToken("diffuseColor"), SdfValueTypeNames->Color3f).Set(color);

    // Create the shader's surface output and connect it to the material's surface output
    UsdShadeOutput shaderOutput = shader.CreateOutput(TfToken("surface"), SdfValueTypeNames->Token);
    material.CreateSurfaceOutput().ConnectToSource(shaderOutput);

    return material;
}

void _createSpheres(const UsdStageRefPtr& stage, const std ::string& name, const Spheres& spheres,
                    const UsdShadeMaterial& material)
{
    // Define a UsdGeomPoints primitive on the stage
    UsdGeomPoints pointsGeom = UsdGeomPoints::Define(stage, SdfPath("/spheres_" + name));

    // Prepare vectors for storing point positions and sizes (radii)
    VtArray<GfVec3f> positions;
    VtArray<float> radii;

    // Loop over the input spheres and store their centers and radii
    for (const Sphere& sphere : spheres)
    {
        positions.push_back(GfVec3f(static_cast<float>(sphere.center.x), static_cast<float>(sphere.center.y),
                                    static_cast<float>(sphere.center.z)));
        radii.push_back(static_cast<float>(sphere.radius)); // Store the radius for each point
    }

    // Set the positions and radii as USD attributes
    pointsGeom.GetPointsAttr().Set(positions);
    pointsGeom.GetWidthsAttr().Set(radii); // Set the widths as the radii of the spheres

    // Bind the material to the points geometry (if needed)
    UsdShadeMaterialBindingAPI(pointsGeom.GetPrim()).Bind(material);
}

// Create cylinders from vector input
void _createCylinders(const UsdStageRefPtr& stage,
                      const std::vector<std::tuple<std::string, GfVec3f, float, float>>& cylinders,
                      const UsdShadeMaterial& material)
{
    for (const auto& cylinderData : cylinders)
    {
        const std::string& name = std::get<0>(cylinderData);
        const GfVec3f& position = std::get<1>(cylinderData);
        float radius = std::get<2>(cylinderData);
        float height = std::get<3>(cylinderData);

        UsdGeomCylinder cylinder = UsdGeomCylinder::Define(stage, SdfPath("/spheres_" + name));
        cylinder.GetRadiusAttr().Set(radius);
        cylinder.GetHeightAttr().Set(height);
        cylinder.AddTranslateOp().Set(position);

        UsdShadeMaterialBindingAPI(cylinder.GetPrim()).Bind(material);
    }
}

// Create cones from vector input
void _createCones(const UsdStageRefPtr& stage, const std::vector<std::tuple<std::string, GfVec3f, float, float>>& cones,
                  const UsdShadeMaterial& material)
{
    for (const auto& coneData : cones)
    {
        const std::string& name = std::get<0>(coneData);
        const GfVec3f& position = std::get<1>(coneData);
        float radius = std::get<2>(coneData);
        float height = std::get<3>(coneData);

        UsdGeomCone cone = UsdGeomCone::Define(stage, SdfPath("/" + name));
        cone.GetRadiusAttr().Set(radius);
        cone.GetHeightAttr().Set(height);
        cone.AddTranslateOp().Set(position);

        UsdShadeMaterialBindingAPI(cone.GetPrim()).Bind(material);
    }
}

// Helper function to clamp the value between 0 and 1 and convert to 8-bit integer
uint8_t FloatToByte(float value)
{
    return static_cast<uint8_t>(std::round(std::fmin(1.0f, std::fmax(0.0f, value)) * 255.0f));
}

// Function to convert RGB floats into a unique uint64_t ID
uint64_t RGBFloatToID(const Vector3d& color)
{
    uint64_t id = 0;

    // Convert float [0.0, 1.0] to 8-bit integer [0, 255] and pack into uint64_t
    id |= static_cast<uint64_t>(FloatToByte(color.x)) << 16; // Red in bits 16-23
    id |= static_cast<uint64_t>(FloatToByte(color.y)) << 8;  // Green in bits 8-15
    id |= static_cast<uint64_t>(FloatToByte(color.z));       // Blue in bits 0-7

    return id;
}
#endif // USE_PIXAR

void USDExporter::exportToFile(const std::string& filename) const
{
#ifdef USE_PIXAR
    UsdStageRefPtr stage = UsdStage::CreateNew(filename);

    std::map<uint64_t, UsdShadeMaterial> materials;
    for (const auto modelDescriptor : _scene.getModelDescriptors())
    {
        auto& model = modelDescriptor->getModel();
        const auto& spheresMap = model.getSpheres();

        for (const auto& spheres : spheresMap)
        {
            const auto material = model.getMaterial(spheres.first);
            const auto& diffuseColor = material->getDiffuseColor();
            const uint64_t index = RGBFloatToID(diffuseColor);

            if (materials.find(index) == materials.end())
                materials[index] = _createSharedMaterial(stage, material->getName(),
                                                         GfVec3f(diffuseColor.x, diffuseColor.y, diffuseColor.z));
        }
    }
    PLUGIN_INFO(1, "Optimized number of materials: " + std::to_string(materials.size()));

    uint64_t progress = 0;
    const auto& modelDescriptors = _scene.getModelDescriptors();
    for (const auto modelDescriptor : modelDescriptors)
    {
        PLUGIN_PROGRESS("- Exporting models", progress, modelDescriptors.size());
        auto& model = modelDescriptor->getModel();
        const auto& spheresMap = model.getSpheres();
        for (const auto& spheres : spheresMap)
        {
            const auto material = model.getMaterial(spheres.first);
            const auto& diffuseColor = material->getDiffuseColor();
            const uint64_t index = RGBFloatToID(diffuseColor);
            // Create a shared material (blue color)
            UsdShadeMaterial m = materials[index];
            _createSpheres(stage, std::to_string(spheres.first), spheres.second, m);
        }
        ++progress;
    }

    stage->GetRootLayer()->Save();
#else
    PLUGIN_THROW("BioExplorer was not compiled with Pixar Universal Scene Description");
#endif
}
} // namespace filesystem
} // namespace io
} // namespace bioexplorer
