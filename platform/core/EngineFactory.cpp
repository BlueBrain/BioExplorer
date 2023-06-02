/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "EngineFactory.h"

#include <platform/core/common/Logs.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>

#if (BRAYNS_USE_OSPRAY)
#include <platform/engines/ospray/OSPRayEngine.h>
#endif

namespace core
{
/** Supported engines */
std::map<std::string, std::string> SUPPORTED_ENGINES = {{ENGINE_OSPRAY, "OSPRayEngine"},
                                                        {ENGINE_OPTIX_6, "OptiX6Engine"},
                                                        {ENGINE_OPTIX_7, "OptiX7Engine"}};

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
        auto createSym = library.getSymbolAddress("brayns_engine_create");
        if (!createSym)
        {
            throw std::runtime_error(std::string("Plugin '") + name + "' is not a valid Brayns engine; missing " +
                                     "brayns_engine_create()");
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
