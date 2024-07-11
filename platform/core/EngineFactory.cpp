/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "EngineFactory.h"

#include <platform/core/common/Logs.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>

#if (PLATFORM_USE_OSPRAY)
#include <platform/engines/ospray/OSPRayEngine.h>
#endif

namespace core
{
/** Supported engines */
std::map<std::string, std::string> SUPPORTED_ENGINES = {{ENGINE_OSPRAY, "OSPRayEngine"},
                                                        {ENGINE_OPTIX_6, "OptiX6Engine"}};

typedef Engine* (*CreateFuncType)(int, const char**, ParametersManager&);

EngineFactory::EngineFactory(const int argc, const char** argv, ParametersManager& parametersManager)
    : _argc{argc}
    , _argv{argv}
    , _parametersManager{parametersManager}
{
}

Engine* EngineFactory::create(const std::string& name)
{
    const auto it = SUPPORTED_ENGINES.find(name);

    if (it == SUPPORTED_ENGINES.end())
        CORE_THROW("Unsupported engine: " + name);

    const auto libraryName = (*it).second;
    if (_engines.count(libraryName) == 0)
        return _loadEngine(libraryName.c_str(), _argc, _argv);
    return _engines[libraryName].get();
}

Engine* EngineFactory::_loadEngine(const std::string& name, int argc, const char* argv[])
{
    try
    {
        DynamicLib library(name);
        auto createSym = library.getSymbolAddress("core_engine_create");
        if (!createSym)
        {
            throw std::runtime_error(std::string("Plugin '") + name + "' is not a valid Core engine; missing " +
                                     "core_engine_create()");
        }

        CreateFuncType createFunc = (CreateFuncType)createSym;
        if (auto plugin = createFunc(argc, argv, _parametersManager))
        {
            _engines.emplace(name, std::unique_ptr<Engine>(plugin));
            _libs.push_back(std::move(library));
            CORE_INFO("Loaded engine '" << name << "'");
            return plugin;
        }
    }
    catch (const std::runtime_error& exc)
    {
        CORE_ERROR("Failed to load engine " << std::quoted(name) << ": " << exc.what());
    }
    return nullptr;
}
} // namespace core
