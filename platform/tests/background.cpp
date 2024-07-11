/*
    Copyright 2019 - 0211 Blue Brain Project / EPFL

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

#include "ClientServer.h"
#include "PDiffHelpers.h"
#include <tests/paths.h>

const std::string SET_ENV_MAP("set-environment-map");

TEST_CASE_FIXTURE(ClientServer, "set_environment_map")
{
    CHECK((makeRequest<core::EnvironmentMapParam, bool>(SET_ENV_MAP, {BRAYNS_TESTDATA_PATH "envmap.jpg"})));

    CHECK(getScene().hasEnvironmentMap());
    getCamera().setPosition({0, 0, 5});
    commitAndRender();
    pdiff::PerceptualDiffParameters parameters;
    parameters.luminance_only = true;
    CHECK(compareTestImage("envmap.png", getFrameBuffer(), parameters));
}

TEST_CASE_FIXTURE(ClientServer, "unset_environment_map")
{
    CHECK((makeRequest<core::EnvironmentMapParam, bool>(SET_ENV_MAP, {""})));

    CHECK(!getScene().hasEnvironmentMap());
    getCamera().setPosition({0, 0, 5});
    commitAndRender();
    CHECK(compareTestImage("no_envmap.png", getFrameBuffer()));
}

TEST_CASE_FIXTURE(ClientServer, "set_invalid_environment_map")
{
    CHECK(!(makeRequest<core::EnvironmentMapParam, bool>(SET_ENV_MAP, {"dont_exists.jpg"})));

    CHECK(!getScene().hasEnvironmentMap());
}
