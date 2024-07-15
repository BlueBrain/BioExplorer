/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

const std::string GET_RENDERER("get-renderer");
const std::string SET_RENDERER("set-renderer");

const std::string GET_RENDERER_PARAMS("get-renderer-params");
const std::string SET_RENDERER_PARAMS("set-renderer-params");

TEST_CASE_FIXTURE(ClientServer, "get_renderer")
{
    auto renderer = makeRequest<core::RenderingParameters>(GET_RENDERER);
    CHECK_EQ(renderer.getCurrentRenderer(), "basic");
}

TEST_CASE_FIXTURE(ClientServer, "get_renderer_params")
{
    CHECK_EQ(getRenderer().getCurrentType(), "basic");
    auto rendererParams = makeRequest<core::PropertyMap>(GET_RENDERER_PARAMS);
    CHECK(rendererParams.getProperties().empty());
}

TEST_CASE_FIXTURE(ClientServer, "change_renderer_from_web_api")
{
    auto& renderer = getRenderer();
    CHECK_EQ(renderer.getCurrentType(), "basic");

    auto params = ClientServer::instance().getBrayns().getParametersManager().getRenderingParameters();
    params.setCurrentRenderer("scivis");
    CHECK((makeRequest<core::RenderingParameters, bool>(SET_RENDERER, params)));
    CHECK_EQ(renderer.getCurrentType(), "scivis");

    core::PropertyMap scivisProps = renderer.getPropertyMap();
    auto rendererParams = makeRequestUpdate<core::PropertyMap>(GET_RENDERER_PARAMS, scivisProps);
    CHECK(!rendererParams.getProperties().empty());
    CHECK_EQ(rendererParams.getProperty<int>("aoSamples"), 1);

    rendererParams.updateProperty("aoSamples", 42);
    CHECK((makeRequest<core::PropertyMap, bool>(SET_RENDERER_PARAMS, rendererParams)));
    CHECK_EQ(renderer.getPropertyMap().getProperty<int>("aoSamples"), 42);

    params.setCurrentRenderer("wrong");
    CHECK_THROWS_AS((makeRequest<core::RenderingParameters, bool>(SET_RENDERER, params)), std::runtime_error);
    CHECK_EQ(renderer.getCurrentType(), "scivis");
}

TEST_CASE_FIXTURE(ClientServer, "update_current_renderer_property")
{
    auto& renderer = getRenderer();
    renderer.setCurrentType("scivis");

    CHECK(renderer.hasProperty("aoWeight"));
    CHECK_EQ(renderer.getProperty<double>("aoWeight"), 0.);

    renderer.updateProperty("aoWeight", 1.5);
    CHECK_EQ(renderer.getProperty<double>("aoWeight"), 1.5);
}
