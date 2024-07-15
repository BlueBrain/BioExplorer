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

#include <plugins/Rockets/jsonSerialization.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("loaderProperties")
{
    core::ModelParams paramsOrig;

    {
        core::PropertyMap properties;
        properties.setProperty({"string", std::string("string")});
        properties.setProperty({"int", 42});
        properties.setProperty({"enum", std::string("b"), {"a", "b", "c", "d"}, {}});
        properties.setProperty({"array", std::array<int, 3>{{1, 2, 3}}});
        paramsOrig.setLoaderProperties(properties);
    }

    const auto jsonStr = to_json(paramsOrig);

    core::ModelParams paramsParse;
    from_json(paramsParse, jsonStr);
    CHECK_EQ(paramsOrig.getLoaderProperties().getProperty<std::string>("string"),
             paramsParse.getLoaderProperties().getProperty<std::string>("string"));
    CHECK_EQ(paramsOrig.getLoaderProperties().getProperty<int32_t>("int"),
             paramsParse.getLoaderProperties().getProperty<int32_t>("int"));
    CHECK_EQ(paramsOrig.getLoaderProperties().getProperty<std::string>("enum"),
             paramsParse.getLoaderProperties().getProperty<std::string>("enum"));

    const auto& origArray = paramsOrig.getLoaderProperties().getProperty<std::array<int, 3>>("array");
    const auto& parseArray = paramsParse.getLoaderProperties().getProperty<std::array<int, 3>>("array");
    CHECK(origArray == parseArray);
}
