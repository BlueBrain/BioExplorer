/* Copyright (c) 2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#include "Covid19Plugin.h"

#include <common/Assembly.h>
#include <common/log.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/pluginapi/Plugin.h>

Covid19Plugin::Covid19Plugin()
    : ExtensionPlugin()
{
}

void Covid19Plugin::init()
{
    auto actionInterface = _api->getActionInterface();
    if (actionInterface)
    {
        PLUGIN_INFO << "Registering 'remove-assembly' endpoint" << std::endl;
        actionInterface->registerRequest<AssemblyDescriptor, Response>(
            "remove-assembly", [&](const AssemblyDescriptor &payload) {
                return _removeAssembly(payload);
            });

        PLUGIN_INFO << "Registering 'add-assembly' endpoint" << std::endl;
        actionInterface->registerRequest<AssemblyDescriptor, Response>(
            "add-assembly", [&](const AssemblyDescriptor &payload) {
                return _addAssembly(payload);
            });

        PLUGIN_INFO << "Registering 'set-protein-color-scheme' endpoint"
                    << std::endl;
        actionInterface->registerRequest<ColorSchemeDescriptor, Response>(
            "set-protein-color-scheme",
            [&](const ColorSchemeDescriptor &payload) {
                return _setColorScheme(payload);
            });

        PLUGIN_INFO << "Registering 'set-protein-amino-acid-sequence' endpoint"
                    << std::endl;
        actionInterface->registerRequest<AminoAcidSequenceDescriptor, Response>(
            "set-protein-amino-acid-sequence",
            [&](const AminoAcidSequenceDescriptor &payload) {
                return _setAminoAcidSequence(payload);
            });

        PLUGIN_INFO << "Registering 'get-protein-amino-acid-sequences' endpoint"
                    << std::endl;
        actionInterface
            ->registerRequest<AminoAcidSequencesDescriptor, Response>(
                "get-protein-amino-acid-sequences",
                [&](const AminoAcidSequencesDescriptor &payload) {
                    return _getAminoAcidSequences(payload);
                });

        PLUGIN_INFO << "Registering 'add-rna-sequence' endpoint" << std::endl;
        actionInterface->registerRequest<RNASequenceDescriptor, Response>(
            "add-rna-sequence", [&](const RNASequenceDescriptor &payload) {
                return _addRNASequence(payload);
            });

        PLUGIN_INFO << "Registering 'add-protein' endpoint" << std::endl;
        actionInterface->registerRequest<ProteinDescriptor, Response>(
            "add-protein", [&](const ProteinDescriptor &payload) {
                return _addProtein(payload);
            });

        PLUGIN_INFO << "Registering 'add-mesh' endpoint" << std::endl;
        actionInterface->registerRequest<MeshDescriptor, Response>(
            "add-mesh",
            [&](const MeshDescriptor &payload) { return _addMesh(payload); });

        PLUGIN_INFO << "Registering 'add-glycans' endpoint" << std::endl;
        actionInterface->registerRequest<GlycansDescriptor, Response>(
            "add-glycans", [&](const GlycansDescriptor &payload) {
                return _addGlycans(payload);
            });
    }
}

Response Covid19Plugin::_removeAssembly(const AssemblyDescriptor &payload)
{
    auto assembly = _assemblies.find(payload.name);
    if (assembly != _assemblies.end())
        _assemblies.erase(assembly);

    return Response();
}

Response Covid19Plugin::_addAssembly(const AssemblyDescriptor &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        AssemblyPtr assembly(new Assembly(scene, payload));
        _assemblies[payload.name] = std::move(assembly);
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
        PLUGIN_ERROR << e.what() << std::endl;
    }
    return response;
}

Response Covid19Plugin::_setColorScheme(const ColorSchemeDescriptor &payload)
{
    Response response;
    auto it = _assemblies.find(payload.assemblyName);
    if (it != _assemblies.end())
        (*it).second->setColorScheme(payload);
    else
    {
        std::stringstream msg;
        msg << "Assembly not found: " << payload.assemblyName;
        PLUGIN_ERROR << msg.str() << std::endl;
        response.status = false;
        response.contents = msg.str();
    }
    return response;
}

Response Covid19Plugin::_setAminoAcidSequence(
    const AminoAcidSequenceDescriptor &payload)
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->setAminoAcidSequence(payload);
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR << msg.str() << std::endl;
            response.status = false;
            response.contents = msg.str();
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response Covid19Plugin::_getAminoAcidSequences(
    const AminoAcidSequencesDescriptor &payload)
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            response.contents = (*it).second->getAminoAcidSequences(payload);
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR << msg.str() << std::endl;
            response.status = false;
            response.contents = msg.str();
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response Covid19Plugin::_addRNASequence(const RNASequenceDescriptor &payload)
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addRNASequence(payload);
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR << msg.str() << std::endl;
            response.status = false;
            response.contents = msg.str();
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response Covid19Plugin::_addProtein(const ProteinDescriptor &payload)
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addProtein(payload);
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR << msg.str() << std::endl;
            response.status = false;
            response.contents = msg.str();
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response Covid19Plugin::_addGlycans(const GlycansDescriptor &payload)
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addGlycans(payload);
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR << msg.str() << std::endl;
            response.status = false;
            response.contents = msg.str();
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response Covid19Plugin::_addMesh(const MeshDescriptor &payload)
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addMesh(payload);
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR << msg.str() << std::endl;
            response.status = false;
            response.contents = msg.str();
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

extern "C" brayns::ExtensionPlugin *brayns_plugin_create(int /*argc*/,
                                                         char ** /*argv*/)
{
    PLUGIN_INFO << "Initializing Covid19 plugin" << std::endl;
    return new Covid19Plugin();
}
