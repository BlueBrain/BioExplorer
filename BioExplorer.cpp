/* Copyright (c) 2020, EPFL/Blue Brain Project
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

#include "BioExplorer.h"

#include <common/Assembly.h>
#include <common/log.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/pluginapi/Plugin.h>

namespace bioexplorer
{
BioExplorer::BioExplorer()
    : ExtensionPlugin()
{
}

void BioExplorer::init()
{
    auto actionInterface = _api->getActionInterface();
    if (actionInterface)
    {
        PLUGIN_INFO << "Registering 'version' endpoint" << std::endl;
        actionInterface->registerRequest<Response>("version", [&]() {
            return _version();
        });

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

        PLUGIN_INFO << "Registering 'apply-transformations' endpoint"
                    << std::endl;
        actionInterface
            ->registerRequest<AssemblyTransformationsDescriptor, Response>(
                "apply-transformations",
                [&](const AssemblyTransformationsDescriptor &payload) {
                    return _applyTransformations(payload);
                });

        PLUGIN_INFO << "Registering 'set-protein-color-scheme' endpoint"
                    << std::endl;
        actionInterface->registerRequest<ColorSchemeDescriptor, Response>(
            "set-protein-color-scheme",
            [&](const ColorSchemeDescriptor &payload) {
                return _setColorScheme(payload);
            });

        PLUGIN_INFO << "Registering "
                       "'set-protein-amino-acid-sequence-as-string' endpoint"
                    << std::endl;
        actionInterface
            ->registerRequest<AminoAcidSequenceAsStringDescriptor, Response>(
                "set-protein-amino-acid-sequence-as-string",
                [&](const AminoAcidSequenceAsStringDescriptor &payload) {
                    return _setAminoAcidSequenceAsString(payload);
                });

        PLUGIN_INFO << "Registering "
                       "'set-protein-amino-acid-sequence-as-range' endpoint"
                    << std::endl;
        actionInterface
            ->registerRequest<AminoAcidSequenceAsRangeDescriptor, Response>(
                "set-protein-amino-acid-sequence-as-range",
                [&](const AminoAcidSequenceAsRangeDescriptor &payload) {
                    return _setAminoAcidSequenceAsRange(payload);
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

Response BioExplorer::_version() const
{
    Response response;
    response.contents = "0.0.1";
    return response;
}

Response BioExplorer::_removeAssembly(const AssemblyDescriptor &payload)
{
    auto assembly = _assemblies.find(payload.name);
    if (assembly != _assemblies.end())
        _assemblies.erase(assembly);

    return Response();
}

Response BioExplorer::_addAssembly(const AssemblyDescriptor &payload)
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

Response BioExplorer::_applyTransformations(
    const AssemblyTransformationsDescriptor &payload) const
{
    Response response;
    auto it = _assemblies.find(payload.assemblyName);
    if (it != _assemblies.end())
        (*it).second->applyTransformations(payload);
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

Response BioExplorer::_setColorScheme(
    const ColorSchemeDescriptor &payload) const
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

Response BioExplorer::_setAminoAcidSequenceAsString(
    const AminoAcidSequenceAsStringDescriptor &payload) const
{
    Response response;
    try
    {
        if (payload.sequence.empty())
            throw std::runtime_error("A valid sequence must be specified");

        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->setAminoAcidSequenceAsString(payload);
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

Response BioExplorer::_setAminoAcidSequenceAsRange(
    const AminoAcidSequenceAsRangeDescriptor &payload) const
{
    Response response;
    try
    {
        if (payload.range.size() != 2)
            throw std::runtime_error("A valid range must be specified");

        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->setAminoAcidSequenceAsRange(payload);
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

Response BioExplorer::_getAminoAcidSequences(
    const AminoAcidSequencesDescriptor &payload) const
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

Response BioExplorer::_addRNASequence(
    const RNASequenceDescriptor &payload) const
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

Response BioExplorer::_addProtein(const ProteinDescriptor &payload) const
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

Response BioExplorer::_addGlycans(const GlycansDescriptor &payload) const
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

Response BioExplorer::_addMesh(const MeshDescriptor &payload) const
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

extern "C" ExtensionPlugin *brayns_plugin_create(int /*argc*/, char ** /*argv*/)
{
    PLUGIN_INFO << "Initializing Covid19 plugin" << std::endl;
    return new BioExplorer();
}
} // namespace bioexplorer
