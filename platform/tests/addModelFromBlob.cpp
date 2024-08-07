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

#include <jsonSerialization.h>

#include <tests/paths.h>

#include "ClientServer.h"

#include <fstream>

const std::string REQUEST_MODEL_UPLOAD("request-model-upload");
const std::string CHUNK("chunk");

TEST_CASE_FIXTURE(ClientServer, "illegal_no_request")
{
    const std::string illegal("illegal");
    getWsClient().sendBinary(illegal.data(), illegal.size());
    process();
    // nothing to test, Core ignores the message and prints a warning
}

TEST_CASE_FIXTURE(ClientServer, "illegal_no_params")
{
    try
    {
        makeRequest<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {});
        REQUIRE(false);
    }
    catch (const rockets::jsonrpc::response_error& e)
    {
        CHECK_EQ(e.code, core::ERROR_ID_MISSING_PARAMS);
        CHECK(e.data.empty());
    }
}

TEST_CASE_FIXTURE(ClientServer, "missing_params")
{
    core::BinaryParam params;
    try
    {
        makeRequest<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});
        REQUIRE(false);
    }
    catch (const rockets::jsonrpc::response_error& e)
    {
        CHECK_EQ(e.code, core::ERROR_ID_MISSING_PARAMS);
        CHECK(e.data.empty());
    }
}

TEST_CASE_FIXTURE(ClientServer, "invalid_size")
{
    core::BinaryParam params;
    params.type = "xyz";
    params.size = 0;
    try
    {
        makeRequest<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});
        REQUIRE(false);
    }
    catch (const rockets::jsonrpc::response_error& e)
    {
        CHECK_EQ(e.code, core::ERROR_ID_MISSING_PARAMS);
        CHECK(e.data.empty());
    }
}

TEST_CASE_FIXTURE(ClientServer, "unsupported_type")
{
    core::BinaryParam params;
    params.type = "blub";
    params.size = 4;
    try
    {
        makeRequest<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});
        REQUIRE(false);
    }
    catch (const rockets::jsonrpc::response_error& e)
    {
        CHECK_EQ(e.code, core::ERROR_ID_UNSUPPORTED_TYPE);
        REQUIRE(e.data.empty());
    }
}

TEST_CASE_FIXTURE(ClientServer, "xyz")
{
    core::BinaryParam params;
    params.size = []
    {
        std::ifstream file(BRAYNS_TESTDATA_MODEL_MONKEY_PATH, std::ios::binary | std::ios::ate);
        return file.tellg();
    }();
    params.type = "xyz";
    params.setPath("monkey.xyz");

    auto request = getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    std::ifstream file(BRAYNS_TESTDATA_MODEL_MONKEY_PATH, std::ios::binary);

    std::vector<char> buffer(1024, 0);

    while (file.read(buffer.data(), buffer.size()))
    {
        const std::streamsize size = file.gcount();
        getWsClient().sendBinary(buffer.data(), size);
        process();
    }

    // read & send last chunk
    const std::streamsize size = file.gcount();
    if (size != 0)
    {
        file.read(buffer.data(), size);
        getWsClient().sendBinary(buffer.data(), size);
    }

    while (!request.is_ready())
        process();
    const auto& model = request.get();
    CHECK_EQ(model.getName(), "monkey");
    CHECK_EQ(model.getPath(), "monkey.xyz");
}

TEST_CASE_FIXTURE(ClientServer, "broken_xyz")
{
    core::BinaryParam params;
    params.size = []
    {
        std::ifstream file(BRAYNS_TESTDATA_MODEL_BROKEN_PATH, std::ios::binary | std::ios::ate);
        return file.tellg();
    }();
    params.type = "xyz";

    auto request = getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    std::ifstream file(BRAYNS_TESTDATA_MODEL_BROKEN_PATH, std::ios::binary);

    std::vector<char> buffer(1024, 0);

    while (file.read(buffer.data(), buffer.size()))
    {
        const std::streamsize size = file.gcount();
        getWsClient().sendBinary(buffer.data(), size);
        process();
    }

    // read & send last chunk
    const std::streamsize size = file.gcount();
    if (size != 0)
    {
        file.read(buffer.data(), size);
        getWsClient().sendBinary(buffer.data(), size);
    }

    while (!request.is_ready())
        process();
    try
    {
        request.get();
    }
    catch (const rockets::jsonrpc::response_error& e)
    {
        CHECK_EQ(e.code, core::ERROR_ID_LOADING_BINARY_FAILED);
        CHECK(std::string(e.what()) == "Invalid content in line 1: 2.500000 3.437500");
    }
}

TEST_CASE_FIXTURE(ClientServer, "cancel")
{
    core::BinaryParam params;
    params.size = 42;
    params.type = "xyz";

    auto request = getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    request.cancel();

    while (!request.is_ready())
        process();

    CHECK_THROWS_AS(request.get(), std::runtime_error);
}

TEST_CASE_FIXTURE(ClientServer, "send_wrong_number_of_bytes")
{
    core::BinaryParam params;
    params.size = 4;
    params.type = "xyz";

    auto request = getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    const std::string wrong("not_four_bytes");
    getWsClient().sendBinary(wrong.data(), wrong.size());

    try
    {
        while (!request.is_ready())
            process();
        request.get();
        REQUIRE(false);
    }
    catch (const rockets::jsonrpc::response_error& e)
    {
        CHECK_EQ(e.code, core::ERROR_ID_INVALID_BINARY_RECEIVE);
    }
}

TEST_CASE_FIXTURE(ClientServer, "cancel_while_loading")
{
    core::BinaryParam params;
    params.size = 4;
    params.type = "forever";

    auto request = getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    const std::string fourBytes("four");
    getWsClient().sendBinary(fourBytes.data(), fourBytes.size());

    process();

    request.cancel();

    while (!request.is_ready())
        process();

    CHECK_THROWS_AS(request.get(), std::runtime_error);
}

TEST_CASE_FIXTURE(ClientServer, "close_client_while_pending_request")
{
    auto wsClient = std::make_unique<rockets::ws::Client>();

    connect(*wsClient);

    core::BinaryParam params;
    params.size = 4;
    params.type = "xyz";

    auto responseFuture = rockets::jsonrpc::Client<rockets::ws::Client>{*wsClient}
                              .request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    auto asyncWait = std::async(std::launch::async,
                                [&responseFuture, &wsClient, this]
                                {
                                    wsClient->process(10);
                                    process();

                                    wsClient.reset(); // close client connection
                                    process();

                                    responseFuture.get();
                                });

    CHECK_THROWS_AS(asyncWait.get(), rockets::jsonrpc::response_error);
}

TEST_CASE_FIXTURE(ClientServer, "obj")
{
    core::BinaryParam params;
    params.size = []
    {
        std::ifstream file(BRAYNS_TESTDATA_MODEL_BENNU_PATH, std::ios::binary | std::ios::ate);
        return file.tellg();
    }();
    params.type = "obj";
    params.setName("bennu");

    auto request = getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {params});

    std::ifstream file(BRAYNS_TESTDATA_MODEL_BENNU_PATH, std::ios::binary);

    std::vector<char> buffer(1024, 0);

    while (file.read(buffer.data(), buffer.size()))
    {
        const std::streamsize size = file.gcount();
        getWsClient().sendBinary(buffer.data(), size);
        process();
    }

    // read & send last chunk
    const std::streamsize size = file.gcount();
    if (size != 0)
    {
        file.read(buffer.data(), size);
        getWsClient().sendBinary(buffer.data(), size);
    }

    while (!request.is_ready())
        process();
    const auto& model = request.get();
    CHECK_EQ(model.getName(), "bennu");
}

TEST_CASE_FIXTURE(ClientServer, "concurrent_requests")
{
    core::BinaryParam xyzParams;
    xyzParams.size = []
    {
        std::ifstream file(BRAYNS_TESTDATA_MODEL_MONKEY_PATH, std::ios::binary | std::ios::ate);
        return file.tellg();
    }();
    xyzParams.type = "xyz";
    xyzParams.chunksID = "coffee";
    xyzParams.setPath("monkey.xyz");

    auto xyzRequest =
        getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {xyzParams});

    std::ifstream xyzFile(BRAYNS_TESTDATA_MODEL_MONKEY_PATH, std::ios::binary);

    ///////////////////

    core::BinaryParam objParams;
    objParams.size = []
    {
        std::ifstream file(BRAYNS_TESTDATA_MODEL_BENNU_PATH, std::ios::binary | std::ios::ate);
        return file.tellg();
    }();
    objParams.type = "obj";
    objParams.chunksID = "1plus1equals2";
    objParams.setName("bennu");

    auto objRequest =
        getJsonRpcClient().request<core::BinaryParam, core::ModelDescriptor>(REQUEST_MODEL_UPLOAD, {objParams});

    std::ifstream objFile(BRAYNS_TESTDATA_MODEL_BENNU_PATH, std::ios::binary);

    ///////////////////

    process();

    std::array<char, 1024> buffer;
    bool xyzDone = false;
    bool objDone = false;
    while (!xyzDone || !objDone)
    {
        if (!xyzDone)
        {
            if (xyzFile.read(buffer.data(), buffer.size()))
            {
                getJsonRpcClient().notify<core::Chunk>(CHUNK, {xyzParams.chunksID});
                const std::streamsize size = xyzFile.gcount();
                getWsClient().sendBinary(buffer.data(), size);
            }
            else
            {
                // read & send last chunk
                const std::streamsize size = xyzFile.gcount();
                if (size != 0)
                {
                    getJsonRpcClient().notify<core::Chunk>(CHUNK, {xyzParams.chunksID});
                    xyzFile.read(buffer.data(), size);
                    getWsClient().sendBinary(buffer.data(), size);
                }
                xyzDone = true;
            }
        }

        if (!objDone)
        {
            if (objFile.read(buffer.data(), buffer.size()))
            {
                getJsonRpcClient().notify<core::Chunk>(CHUNK, {objParams.chunksID});
                const std::streamsize size = objFile.gcount();
                getWsClient().sendBinary(buffer.data(), size);
            }
            else
            {
                // read & send last chunk
                const std::streamsize size = objFile.gcount();
                if (size != 0)
                {
                    getJsonRpcClient().notify<core::Chunk>(CHUNK, {objParams.chunksID});
                    objFile.read(buffer.data(), size);
                    getWsClient().sendBinary(buffer.data(), size);
                }
                objDone = true;
            }
        }
        process();
    }

    while (!xyzRequest.is_ready() || !objRequest.is_ready())
        process();

    const auto& xyzModel = xyzRequest.get();
    CHECK_EQ(xyzModel.getName(), "monkey");
    CHECK_EQ(xyzModel.getPath(), "monkey.xyz");

    const auto& objModel = objRequest.get();
    CHECK_EQ(objModel.getName(), "bennu");
}
