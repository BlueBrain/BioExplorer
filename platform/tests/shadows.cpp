/*
    Copyright 2018 - 0211 Blue Brain Project / EPFL

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

#include <platform/core/Core.h>
#include <tests/paths.h>

#include <platform/core/common/Types.h>
#include <platform/core/common/light/Light.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Renderer.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "PDiffHelpers.h"

TEST_CASE("cylinders")
{
    std::vector<const char*> argv = {{RENDERER_PROPERTY_SHADOW_INTENSITY.name, "--disable-accumulation",
                                      "--window-size", "1600", "900", "--plugin", "braynsCircuitViewer",
                                      "--no-head-light"}};

    core::Core core(argv.size(), argv.data());
    core.getParametersManager().getRenderingParameters().setCurrentRenderer("advanced_simulation");
    core.commit();
    auto& scene = core.getEngine().getScene();

    auto model = scene.createModel();

    const core::Vector3f WHITE = {1.f, 1.f, 1.f};
    const core::Vector3f RED = {1.f, 0.f, 0.f};
    const core::Vector3f GREEN = {0.f, 1.f, 0.f};

    constexpr size_t materialIdRed = 0;
    auto materialRed = model->createMaterial(materialIdRed, "Cylinders Red");
    materialRed->setDiffuseColor(RED);
    materialRed->setSpecularColor(RED);
    materialRed->setSpecularExponent(10.f);

    constexpr size_t materialIdGreen = 1;
    auto materialGreen = model->createMaterial(materialIdGreen, "Cylinders Green");
    materialGreen->setDiffuseColor(GREEN);
    materialGreen->setSpecularColor(GREEN);
    materialGreen->setSpecularExponent(10.f);
    materialGreen->setOpacity(0.25f);

    constexpr float row_stride = 0.2f;
    constexpr float thickness = 0.05f;
    constexpr float height = 0.35f;
    constexpr size_t num_cols = 7;

    for (size_t col = 0; col < num_cols; ++col)
    {
        const bool odd = col % 2;
        const size_t num_rows = odd ? num_cols - 1 : num_cols;
        const size_t materialId = odd ? materialIdGreen : materialIdRed;

        for (size_t row = 0; row < num_rows; ++row)
        {
            const float start_x = -(static_cast<float>(num_rows - 1) * row_stride * 0.5f);
            const float start_z = -(static_cast<float>(num_cols - 1) * row_stride * 0.5f);

            const float x = start_x + row * row_stride;
            const float z = start_z + col * row_stride;
            model->addCylinder(materialId, {{x, 0.0f, z}, {x, height, z}, thickness});
        }
    }

    auto modelDesc = std::make_shared<core::ModelDescriptor>(std::move(model), "Cylinders");
    scene.addModel(modelDesc);
    scene.getLightManager().clearLights();

    scene.getLightManager().addLight(
        std::make_unique<core::DirectionalLight>(core::Vector3f(0.f, 0.f, -1.f), 0, WHITE, 1.0f, true));
    scene.commitLights();

    auto& camera = core.getEngine().getCamera();

    const double dist = 1.5;
    const double alpha = -std::atan(2 * height / dist);
    const core::Quaterniond orientation(glm::angleAxis(alpha, core::Vector3d(1.0, 0.0, 0.0)));
    camera.setOrientation(orientation);
    camera.setPosition(core::Vector3f(0.f, 2 * height, dist));

    auto& renderer = core.getEngine().getRenderer();
    renderer.updateProperty(RENDERER_PROPERTY_SHADOW_INTENSITY.name.c_str(), 1.);

    core.commitAndRender();
    CHECK(compareTestImage("shadowCylinders.png", core.getEngine().getFrameBuffer()));
}
