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

using Vec2 = std::array<unsigned, 2>;
const Vec2 vecVal{{1, 1}};

TEST_CASE("plugin_actions")
{
    ClientServer clientServer({"--plugin", "myPlugin"});

    makeNotification("notify");
    core::PropertyMap input;
    input.setProperty({"value", 42});
    makeNotification("notify-param", input);

    // wrong input, cannot test though
    makeNotification("notify-param", vecVal);

    core::PropertyMap output;
    output.setProperty({"result", false});
    auto result = makeRequestUpdate("request", output);
    CHECK(result.getProperty<bool>("result"));

    result = makeRequestUpdate("request-param", input, output);
    CHECK(result.getProperty<bool>("result"));

    // wrong input
    CHECK_THROWS_AS(makeRequestUpdate("request-param", vecVal, output), std::runtime_error);

    makeNotification("hello");
    makeNotification("foo", vecVal);
    CHECK_EQ(makeRequest<std::string>("who"), "me");
    CHECK((makeRequest<Vec2, Vec2>("echo", vecVal) == vecVal));

    clientServer.getBrayns().getParametersManager().getRenderingParameters().setCurrentRenderer("myrenderer");
    clientServer.getBrayns().commitAndRender();

    auto props = clientServer.getBrayns().getEngine().getRenderer().getPropertyMap();
    CHECK(props.hasProperty("awesome"));
    CHECK_EQ(props.getProperty<int>("awesome"), 42);

    props.updateProperty("awesome", 10);

    CHECK((makeRequest<core::PropertyMap, bool>("set-renderer-params", props)));

    const auto& newProps = clientServer.getBrayns().getEngine().getRenderer().getPropertyMap();
    CHECK_EQ(newProps.getProperty<int>("awesome"), 10);
}
