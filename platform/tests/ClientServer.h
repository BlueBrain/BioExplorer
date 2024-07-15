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

#pragma once

#include <platform/core/Core.h>
#include <platform/core/common/loader/Loader.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>

#include <plugins/Rockets/jsonSerialization.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <rockets/jsonrpc/client.h>
#include <rockets/ws/client.h>

#include <future>

template <typename T>
bool is_ready(const std::future<T>& f)
{
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

const size_t CLIENT_PROCESS_TIMEOUT = 5;  /*ms*/
const size_t SERVER_PROCESS_RETRIES = 10; /*ms*/

class ForeverLoader : public core::Loader
{
public:
    using core::Loader::Loader;

    bool isSupported(const std::string& storage, const std::string& extension) const final
    {
        return filename == "forever";
    }

    std::vector<std::string> getSupportedStorage() const { return {}; }
    std::string getName() const { return "forever"; }
    core::PropertyMap getProperties() const { return {}; }
    core::ModelDescriptorPtr importFromBlob(core::Blob&&, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final
    {
        for (;;)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            callback.updateProgress("still not done", 0.f);
        }
        return {};
    }

    core::ModelDescriptorPtr importFromFile(const std::string&, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final
    {
        for (;;)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            callback.updateProgress("still not done", 0.f);
        }
        return {};
    }
};

class ClientServer
{
public:
    static ClientServer& instance()
    {
        if (!_instance)
            throw std::runtime_error("Could not initialize client/server");
        return *_instance;
    }

    ClientServer(std::vector<const char*> additionalArgv = {"demo"})
        : _wsClient{std::make_unique<rockets::ws::Client>()}
        , _client(*_wsClient)
    {
        std::vector<const char*> argv{"core", "--http-server", "localhost:0"};
        for (const auto& arg : additionalArgv)
            argv.push_back(arg);
        const int argc = argv.size();
        _core.reset(new core::Core(argc, argv.data()));
        _core->getParametersManager().getApplicationParameters().setImageStreamFPS(0);
        _core->commitAndRender();

        auto& scene = _core->getEngine().getScene();
        scene.getLoaderRegistry().registerLoader(std::make_unique<ForeverLoader>(scene));

        connect(*_wsClient);
        _instance = this;
    }

    ~ClientServer()
    {
        _wsClient.reset();
        _core->commit(); // handle disconnect of client
    }

    void connect(rockets::ws::Client& client)
    {
        const auto uri = _core->getParametersManager().getApplicationParameters().getHttpServerURI();

        auto connectFuture = client.connect("ws://" + uri, "rockets");
        while (!is_ready(connectFuture))
        {
            client.process(CLIENT_PROCESS_TIMEOUT);
            _core->commit();
        }
        connectFuture.get();
    }

    template <typename Params, typename RetVal>
    RetVal makeRequest(const std::string& method, const Params& params)
    {
        auto request = _client.request<Params, RetVal>(method, params);
        while (!request.is_ready())
        {
            _wsClient->process(0);
            _core->commit();
        }

        return request.get();
    }

    template <typename RetVal>
    RetVal makeRequest(const std::string& method, const std::string& params)
    {
        auto request = _client.request(method, params);
        while (!request.is_ready())
        {
            _wsClient->process(0);
            _core->commit();
        }

        RetVal retVal;
        ::from_json(retVal, request.get().result);
        return retVal;
    }

    template <typename RetVal>
    RetVal makeRequest(const std::string& method)
    {
        auto request = _client.request<RetVal>(method);
        while (!request.is_ready())
        {
            _wsClient->process(0);
            _core->commit();
        }

        return request.get();
    }

    template <typename Params>
    std::string makeRequestJSONReturn(const std::string& method, const Params& params)
    {
        auto request = _client.request(method, to_json(params));
        while (!request.is_ready())
        {
            _wsClient->process(0);
            _core->commit();
        }

        return request.get().result;
    }

    template <typename Params, typename RetVal>
    RetVal makeRequestUpdate(const std::string& method, const Params& params, RetVal baseObject)
    {
        auto promise = std::make_shared<std::promise<RetVal>>();
        auto callback = [promise, &baseObject](auto response)
        {
            if (response.isError())
                promise->set_exception(std::make_exception_ptr(rockets::jsonrpc::response_error(response.error)));
            else
            {
                if (!from_json(baseObject, response.result))
                    promise->set_exception(std::make_exception_ptr(
                        rockets::jsonrpc::response_error("Response JSON conversion failed",
                                                         rockets::jsonrpc::ErrorCode::invalid_json_response)));
                else
                    promise->set_value(std::move(baseObject));
            }
        };

        _client.request(method, to_json(params), callback);
        auto future = promise->get_future();

        while (!rockets::is_ready(future))
        {
            _wsClient->process(0);
            _core->commit();
        }

        return future.get();
    }

    template <typename RetVal>
    RetVal makeRequestUpdate(const std::string& method, RetVal baseObject)
    {
        auto promise = std::make_shared<std::promise<RetVal>>();
        auto callback = [promise, &baseObject](auto response)
        {
            if (response.isError())
                promise->set_exception(std::make_exception_ptr(rockets::jsonrpc::response_error(response.error)));
            else
            {
                if (!from_json(baseObject, response.result))
                    promise->set_exception(std::make_exception_ptr(
                        rockets::jsonrpc::response_error("Response JSON conversion failed",
                                                         rockets::jsonrpc::ErrorCode::invalid_json_response)));
                else
                    promise->set_value(std::move(baseObject));
            }
        };

        _client.request(method, "", callback);
        auto future = promise->get_future();

        while (!rockets::is_ready(future))
        {
            _wsClient->process(0);
            _core->commit();
        }

        return future.get();
    }

    template <typename Params>
    void makeNotification(const std::string& method, const Params& params)
    {
        _client.notify<Params>(method, params);

        _wsClient->process(CLIENT_PROCESS_TIMEOUT);
        for (size_t i = 0; i < SERVER_PROCESS_RETRIES; ++i)
            _core->commit();
    }

    void makeNotification(const std::string& method)
    {
        _client.notify(method, std::string());

        _wsClient->process(CLIENT_PROCESS_TIMEOUT);
        for (size_t i = 0; i < SERVER_PROCESS_RETRIES; ++i)
            _core->commit();
    }

    auto& getBrayns() { return *_core; }
    auto& getWsClient() { return *_wsClient; }
    auto& getJsonRpcClient() { return _client; }
    void process()
    {
        _wsClient->process(CLIENT_PROCESS_TIMEOUT);
        _core->commit();
    }

private:
    static ClientServer* _instance;
    std::unique_ptr<core::Core> _core;
    std::unique_ptr<rockets::ws::Client> _wsClient;
    rockets::jsonrpc::Client<rockets::ws::Client> _client;
};

class Client
{
private:
    rockets::ws::Client _wsClient;

public:
    rockets::jsonrpc::Client<rockets::ws::Client> client{_wsClient};

    Client(ClientServer& server) { server.connect(_wsClient); }
    void process() { _wsClient.process(CLIENT_PROCESS_TIMEOUT); }
};

ClientServer* ClientServer::_instance{nullptr};

template <typename Params, typename RetVal>
RetVal makeRequest(const std::string& method, const Params& params)
{
    return ClientServer::instance().makeRequest<Params, RetVal>(method, params);
}

template <typename RetVal>
RetVal makeRequest(const std::string& method, const std::string& params)
{
    return ClientServer::instance().makeRequest<RetVal>(method, params);
}

template <typename RetVal>
RetVal makeRequest(const std::string& method)
{
    return ClientServer::instance().makeRequest<RetVal>(method);
}

template <typename Params>
std::string makeRequestJSONReturn(const std::string& method, const Params& params)
{
    return ClientServer::instance().makeRequestJSONReturn<Params>(method, params);
}

template <typename Params, typename RetVal>
RetVal makeRequestUpdate(const std::string& method, const Params& params, RetVal baseObject)
{
    return ClientServer::instance().makeRequestUpdate<Params, RetVal>(method, params, baseObject);
}

template <typename RetVal>
RetVal makeRequestUpdate(const std::string& method, RetVal baseObject)
{
    return ClientServer::instance().makeRequestUpdate<RetVal>(method, baseObject);
}

template <typename Params>
void makeNotification(const std::string& method, const Params& params)
{
    ClientServer::instance().makeNotification<Params>(method, params);
}

void makeNotification(const std::string& method)
{
    ClientServer::instance().makeNotification(method);
}

core::Camera& getCamera()
{
    return ClientServer::instance().getBrayns().getEngine().getCamera();
}

core::Scene& getScene()
{
    return ClientServer::instance().getBrayns().getEngine().getScene();
}

core::Renderer& getRenderer()
{
    return ClientServer::instance().getBrayns().getEngine().getRenderer();
}

auto& getFrameBuffer()
{
    return ClientServer::instance().getBrayns().getEngine().getFrameBuffer();
}

auto& getWsClient()
{
    return ClientServer::instance().getWsClient();
}

auto& getJsonRpcClient()
{
    return ClientServer::instance().getJsonRpcClient();
}

void process()
{
    ClientServer::instance().process();
}

void connect(rockets::ws::Client& client)
{
    ClientServer::instance().connect(client);
}

void commitAndRender()
{
    ClientServer::instance().getBrayns().commitAndRender();
}
