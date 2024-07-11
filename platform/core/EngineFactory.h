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

#pragma once

#include <platform/core/common/Types.h>
#include <platform/core/common/utils/DynamicLib.h>

namespace core
{
/**
 * The engine factory is in charge of instantiating engines according to their
 * name (ospray, optix or firerays). If Core does not find the 3rd party
 * library at compilation time, the according lib is not generated and the
 * get method returns a null pointer.
 */
class EngineFactory
{
public:
    /**
     * @brief Constructor
     * @param argc Number of command line arguments
     * @param argv Command line arguments
     * @param parametersManager Container for all parameters (application,
     *        rendering, geometry, scene)
     */
    EngineFactory(int argc, const char** argv, ParametersManager& parametersManager);

    /**
     * @brief Create an instance of the engine corresponding the given name. If
     *        the name is incorrect, a null pointer is returned.
     * @param name of the engine library, e.g. OSPRayEngine
     * @return A pointer to the engine, null if the engine could not be
     *         instantiated
     */
    Engine* create(const std::string& name);

private:
    int _argc;
    const char** _argv;
    ParametersManager& _parametersManager;
    std::vector<DynamicLib> _libs;
    std::map<std::string, std::unique_ptr<Engine>> _engines;

    Engine* _loadEngine(const std::string& name, int argc, const char* argv[]);
};
} // namespace core
