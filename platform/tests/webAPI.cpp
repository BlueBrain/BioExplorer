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

#include <jsonPropertyMap.h>

#include "ClientServer.h"
#include <platform/core/engineapi/Renderer.h>

TEST_CASE_FIXTURE(ClientServer, "change_fov")
{
    core::PropertyMap cameraParams;
    cameraParams.setProperty({CAMERA_PROPERTY_FOVY, 10., .1, 360.});
    CHECK((makeRequest<core::PropertyMap, bool>("set-camera-params", cameraParams)));
}

TEST_CASE_FIXTURE(ClientServer, "reset_camera")
{
    const auto& orientation = getCamera().getOrientation();
    getCamera().setOrientation({0, 0, 0, 1});
    makeNotification("reset-camera");
    CHECK_EQ(getCamera().getOrientation(), orientation);
}

TEST_CASE_FIXTURE(ClientServer, "inspect")
{
    auto inspectResult = makeRequest<std::array<double, 2>, core::Renderer::PickResult>("inspect", {{0.5, 0.5}});
    CHECK(inspectResult.hit);
    CHECK(glm::all(glm::epsilonEqual(inspectResult.pos, {0.5, 0.5, 1.19209289550781e-7}, 0.000001)));

    auto failedInspectResult = makeRequest<std::array<double, 2>, core::Renderer::PickResult>("inspect", {{10, -10}});
    CHECK(!failedInspectResult.hit);
}

TEST_CASE_FIXTURE(ClientServer, "schema_non_existing_endpoint")
{
    CHECK_THROWS_AS((makeRequest<core::SchemaParam, std::string>("schema", {"foo"})), std::runtime_error);
}

TEST_CASE_FIXTURE(ClientServer, "schema")
{
    std::string result = makeRequestJSONReturn<core::SchemaParam>("schema", core::SchemaParam{"camera"});

    using namespace rapidjson;
    Document json(kObjectType);
    json.Parse(result.c_str());
    CHECK(json.HasMember("title"));
}
