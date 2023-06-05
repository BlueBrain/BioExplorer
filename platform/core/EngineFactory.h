/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#ifndef ENGINEFACTORY_H
#define ENGINEFACTORY_H

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

#endif // ENGINEFACTORY_H
