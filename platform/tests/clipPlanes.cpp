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

#include "ClientServer.h"

const std::string ADD_CLIP_PLANE("add-clip-plane");
const std::string GET_CLIP_PLANES("get-clip-planes");
const std::string REMOVE_CLIP_PLANES("remove-clip-planes");
const std::string UPDATE_CLIP_PLANE("update-clip-plane");

TEST_CASE_FIXTURE(ClientServer, "add_plane")
{
    REQUIRE(getScene().getClipPlanes().empty());
    const core::Plane equation{{1.0, 2.0, 3.0, 4.0}};
    const auto result = makeRequest<core::Plane, core::ClipPlane>(ADD_CLIP_PLANE, equation);
    CHECK_EQ(result.getID(), 0);
    CHECK(result.getPlane() == equation);
    REQUIRE_EQ(getScene().getClipPlanes().size(), 1);
    CHECK(getScene().getClipPlane(0)->getPlane() == equation);
    CHECK(getScene().getClipPlane(1) == core::ClipPlanePtr());

    getScene().removeClipPlane(0);
    CHECK(getScene().getClipPlanes().empty());
}

TEST_CASE_FIXTURE(ClientServer, "get_planes")
{
    const core::Plane equation1{{1.0, 1.0, 1.0, 1.0}};
    const core::Plane equation2{{2.0, 2.0, 2.0, 2.0}};

    const auto id1 = getScene().addClipPlane(equation1);
    const auto id2 = getScene().addClipPlane(equation2);

    const auto result = makeRequest<core::ClipPlanes>(GET_CLIP_PLANES);
    CHECK_EQ(result.size(), 2);
    CHECK(result[0]->getPlane() == equation1);
    CHECK(result[1]->getPlane() == equation2);

    getScene().removeClipPlane(id1);
    getScene().removeClipPlane(id2);
}

TEST_CASE_FIXTURE(ClientServer, "update_plane")
{
    Client client(ClientServer::instance());

    const core::Plane equation1{{1.0, 1.0, 1.0, 1.0}};
    const core::Plane equation2{{2.0, 2.0, 2.0, 2.0}};

    const auto id1 = getScene().addClipPlane(equation1);

    makeRequest<core::ClipPlane, bool>(UPDATE_CLIP_PLANE, core::ClipPlane(id1, equation2));

    CHECK(getScene().getClipPlane(id1)->getPlane() == equation2);
    getScene().removeClipPlane(id1);
}

TEST_CASE_FIXTURE(ClientServer, "remove_planes")
{
    const core::Plane equation{{1.0, 2.0, 3.0, 4.0}};
    const auto id1 = getScene().addClipPlane(equation);
    const auto id2 = getScene().addClipPlane(equation);
    const auto id3 = getScene().addClipPlane(equation);
    makeRequest<size_ts, bool>(REMOVE_CLIP_PLANES, {id2});
    makeRequest<size_ts, bool>(REMOVE_CLIP_PLANES, {id1, id3});
    CHECK(getScene().getClipPlanes().empty());
}

TEST_CASE_FIXTURE(ClientServer, "notifications")
{
    Client client(ClientServer::instance());

    bool called = false;
    core::ClipPlane notified;
    size_ts ids;
    client.client.connect<core::ClipPlane>(UPDATE_CLIP_PLANE,
                                           [&notified, &called](const core::ClipPlane& plane)
                                           {
                                               notified = plane;
                                               called = true;
                                           });
    client.client.connect<size_ts>(REMOVE_CLIP_PLANES,
                                   [&ids, &called](const size_ts& ids_)
                                   {
                                       ids = ids_;
                                       called = true;
                                   });
    process();

    auto added = makeRequest<core::Plane, core::ClipPlane>(ADD_CLIP_PLANE, {{1.0, 1.0, 1.0, 1.0}});

    process();
    for (size_t attempts = 0; attempts != 100 && !called; ++attempts)
        client.process();
    REQUIRE(called);

    CHECK_EQ(notified.getID(), added.getID());
    CHECK(notified.getPlane() == added.getPlane());

    added.setPlane({{2.0, 2.0, 2.0, 2.0}});
    makeRequest<core::ClipPlane, bool>(UPDATE_CLIP_PLANE, added);
    notified = core::ClipPlane();

    process();
    called = false;
    for (size_t attempts = 0; attempts != 100 && !called; ++attempts)
        client.process();
    REQUIRE(called);

    CHECK_EQ(notified.getID(), added.getID());
    CHECK(notified.getPlane() == added.getPlane());

    makeRequest<size_ts, bool>(REMOVE_CLIP_PLANES, {added.getID()});

    process();
    called = false;
    for (size_t attempts = 0; attempts != 100 && !called; ++attempts)
        client.process();
    REQUIRE(called);

    CHECK(ids == size_ts{added.getID()});
}
