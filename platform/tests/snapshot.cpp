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

#include <jsonSerialization.h>
#include <tests/paths.h>

#include "ClientServer.h"
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Renderer.h>

#include "tests/PDiffHelpers.h"

TEST_CASE_FIXTURE(ClientServer, "snapshot")
{
    core::SnapshotParams params;
    params.format = "png";
    params.size = {50, 50};
    params.quality = 90;

    auto image = makeRequest<core::SnapshotParams, core::ImageGenerator::ImageBase64>("snapshot", params);

    CHECK(compareBase64TestImage(image, "snapshot.png"));
}

TEST_CASE_FIXTURE(ClientServer, "snapshot_with_render_params")
{
    // move far enough away to see the background
    auto camera{std::make_unique<core::Camera>()};
    *camera = getCamera();
    camera->setPosition({0, 0, 50});

    core::SnapshotParams params;
    params.camera = std::move(camera);
    params.format = "jpg";
    params.size = {5, 5};
    params.quality = 75;
    params.name = "black_image";

    auto image = makeRequest<core::SnapshotParams, core::ImageGenerator::ImageBase64>("snapshot", params);

    // use a red background, as opposed to the default black
    auto renderingParams{std::make_unique<core::RenderingParameters>()};
    renderingParams->setBackgroundColor({1, 0, 0});
    params.renderingParams = std::move(renderingParams);
    params.name = "red_image";
    auto image_with_red_background =
        makeRequest<core::SnapshotParams, core::ImageGenerator::ImageBase64>("snapshot", params);

    CHECK_NE(image.data, image_with_red_background.data);
}

TEST_CASE_FIXTURE(ClientServer, "snapshot_empty_params")
{
    CHECK_THROWS_AS((makeRequest<core::SnapshotParams, core::ImageGenerator::ImageBase64>("snapshot",
                                                                                          core::SnapshotParams())),
                    rockets::jsonrpc::response_error);
}

TEST_CASE_FIXTURE(ClientServer, "snapshot_illegal_format")
{
    core::SnapshotParams params;
    params.size = {5, 5};
    params.format = "";
    CHECK_THROWS_AS((makeRequest<core::SnapshotParams, core::ImageGenerator::ImageBase64>("snapshot", params)),
                    rockets::jsonrpc::response_error);
}
