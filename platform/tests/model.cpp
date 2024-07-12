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

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include <jsonPropertyMap.h>
#include <jsonSerialization.h>

#include "ClientServer.h"

const std::string GET_INSTANCES("get-instances");
const std::string REMOVE_MODEL("remove-model");
const std::string UPDATE_INSTANCE("update-instance");
const std::string UPDATE_MODEL("update-model");
const std::string SET_PROPERTIES("set-model-properties");
const std::string GET_PROPERTIES("get-model-properties");
const std::string MODEL_PROPERTIES_SCHEMA("model-properties-schema");

TEST_CASE_FIXTURE(ClientServer, "set_properties")
{
    auto model = getScene().getModel(0);

    core::PropertyMap props;
    props.setProperty({"bla", 0});
    model->setProperties(props);

    core::PropertyMap propsNew;
    propsNew.setProperty({"bla", 42});
    CHECK((makeRequest<core::ModelProperties, bool>(SET_PROPERTIES, {model->getModelID(), propsNew})));

    CHECK_EQ(model->getProperties().getProperty<int32_t>("bla"), 42);

    SUBCASE("get_model_properties_schema")
    {
        auto result = makeRequestJSONReturn<core::ObjectID>(MODEL_PROPERTIES_SCHEMA, {model->getModelID()});

        using namespace rapidjson;
        Document json(kObjectType);
        json.Parse(result.c_str());
        CHECK(json["properties"].HasMember("bla"));
    }

    SUBCASE("get_model_properties")
    {
        auto result =
            makeRequestUpdate<core::ObjectID, core::PropertyMap>(GET_PROPERTIES, {model->getModelID()}, props);
        REQUIRE(result.hasProperty("bla"));
        CHECK_EQ(result.getProperty<int32_t>("bla"), 42);
    }
}

TEST_CASE_FIXTURE(ClientServer, "update_model")
{
    auto model = getScene().getModel(0);

    const uint32_t id = model->getModelID();

    model->setBoundingBox(true); // different from default

    CHECK(model->getVisible());
    CHECK(model->getBoundingBox());

    // create partial model description to only update visible state
    using namespace rapidjson;
    Document json(kObjectType);
    json.AddMember("id", id, json.GetAllocator());
    json.AddMember("visible", false, json.GetAllocator());
    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    json.Accept(writer);

    CHECK((makeRequest<bool>(UPDATE_MODEL, buffer.GetString())));

    CHECK(!model->getVisible());
    CHECK(model->getBoundingBox()); // shall remain untouched
}

TEST_CASE_FIXTURE(ClientServer, "instances")
{
    auto model = getScene().getModel(0);

    core::Transformation trafo;
    trafo.setTranslation({10, 200, -400});
    model->addInstance({true, false, trafo});
    process();

    SUBCASE("add_instance")
    {
        REQUIRE_EQ(model->getInstances().size(), 2);
        auto& instance = model->getInstances()[1];
        CHECK(instance.getTransformation() == trafo);
        CHECK(instance.getVisible());
        CHECK(!instance.getBoundingBox());

        const auto& instances =
            makeRequest<core::GetInstances, core::ModelInstances>(GET_INSTANCES, {model->getModelID(), {1, 2}});
        REQUIRE_EQ(instances.size(), 1);
        auto& rpcInstance = instances[0];
        CHECK(rpcInstance.getTransformation() == trafo);
        CHECK(rpcInstance.getVisible());
        CHECK(!rpcInstance.getBoundingBox());
    }

    SUBCASE("update_instance")
    {
        REQUIRE_EQ(model->getInstances().size(), 2);
        auto instance = model->getInstances()[1];

        instance.setBoundingBox(true);
        instance.setVisible(false);
        auto scaleTrafo = instance.getTransformation();
        scaleTrafo.setScale({1, 2, 3});
        instance.setTransformation(scaleTrafo);

        CHECK(makeRequest<core::ModelInstance, bool>(UPDATE_INSTANCE, instance));

        process();

        const auto& updatedInstance = model->getInstances()[1];
        CHECK(updatedInstance.getTransformation() == scaleTrafo);
        CHECK(!updatedInstance.getVisible());
        CHECK(updatedInstance.getBoundingBox());
    }

    SUBCASE("remove_instance")
    {
        REQUIRE_EQ(model->getInstances().size(), 2);

        model->removeInstance(model->getInstances()[1].getInstanceID());
        process();

        REQUIRE_EQ(model->getInstances().size(), 1);
    }
}

TEST_CASE_FIXTURE(ClientServer, "remove_model")
{
    const auto desc = getScene().getModel(0);

    CHECK(makeRequest<size_ts, bool>(REMOVE_MODEL, {desc->getModelID()}));

    CHECK_EQ(getScene().getNumModels(), 0);
}
