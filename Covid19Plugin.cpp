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

#include <common/Protein.h>
#include <common/RNASequence.h>
#include <common/log.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/common/Progress.h>
#include <brayns/common/utils/enumUtils.h>
#include <brayns/common/utils/utils.h>
#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/io/MeshLoader.h>
#include <brayns/parameters/ParametersManager.h>
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
        PLUGIN_INFO << "Registering 'build-structure' endpoint" << std::endl;
        actionInterface->registerRequest<StructureDescriptor, Response>(
            "build-structure", [&](const StructureDescriptor &payload) {
                return _buildStructure(payload);
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

        PLUGIN_INFO << "Registering 'load-rna-sequence' endpoint" << std::endl;
        actionInterface->registerRequest<RNADescriptor, Response>(
            "load-rna-sequence",
            [&](const RNADescriptor &payload) { return _loadRNA(payload); });
    }
}

Response Covid19Plugin::_buildStructure(const StructureDescriptor &payload)
{
    Response response;
    PLUGIN_INFO << "Initializing structure from " << payload.path << std::endl;
    PLUGIN_INFO << "Number of instances: " << payload.occurrences << std::endl;
    PLUGIN_INFO << "Virus radius    : " << payload.assemblyRadius << std::endl;

    try
    {
        // Checks
        if (payload.upVector.size() != 3)
            throw std::runtime_error("Invalid up vector");

        auto &scene = _api->getScene();
        const std::string ext = brayns::extractExtension(payload.path);

        brayns::ModelDescriptorPtr modelDescriptor{nullptr};

        if (ext == "pdb" || ext == "pdb1")
        {
            ProteinPtr protein(new Protein(scene, payload.name, payload.path,
                                           payload.atomRadiusMultiplier));
            modelDescriptor = protein->getModelDescriptor();
            _proteins[payload.path] = std::move(protein);
        }
        else if (ext == "obj")
        {
            const auto loader = brayns::MeshLoader(scene);
            modelDescriptor =
                loader.importFromFile(payload.path, brayns::LoaderProgress(),
                                      brayns::PropertyMap());
        }
        else
        {
            response.status = false;
            response.contents = "Unsupported file format";
            return response;
        }

        scene.addModel(modelDescriptor);
        auto &model = modelDescriptor->getModel();
        const auto &bounds = model.getBounds();
        const brayns::Vector3f &center = bounds.getCenter();

        const float offset = 2.f / payload.occurrences;
        const float increment = M_PI * (3.f - sqrt(5.f));
        srand(time(NULL));
        size_t rnd = payload.randomize ? rand() % payload.occurrences : 1;

        size_t instanceCount = 0;
        for (size_t i = 0; i < payload.occurrences; ++i)
        {
            const float y = ((i * offset) - 1.f) + (offset / 2.f);
            const float r = sqrt(1.f - pow(y, 2.f));
            const float phi = ((i + rnd) % payload.occurrences) * increment;
            const float x = cos(phi) * r;
            const float z = sin(phi) * r;
            const auto direction = brayns::Vector3f(x, y, z);

            if (payload.halfStructure &&
                (direction.x > 0.f && direction.y > 0.f && direction.z > 0.f))
                continue;

            brayns::Transformation tf;
            const brayns::Vector3f position =
                payload.assemblyRadius * direction;
            tf.setTranslation(position - center);
            tf.setRotationCenter(center);

            const brayns::Vector3f up = {payload.upVector[0],
                                         payload.upVector[1],
                                         payload.upVector[2]};
            tf.setRotation(glm::quatLookAt(direction, up));

            if (instanceCount == 0)
                modelDescriptor->setTransformation(tf);
            else
            {
                brayns::ModelInstance instance(true, false, tf);
                modelDescriptor->addInstance(instance);
            }
            ++instanceCount;
        }

        PLUGIN_INFO << "Structure successfully built" << std::endl;
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response Covid19Plugin::_setColorScheme(const ColorSchemeDescriptor &payload)
{
    Response response;
    auto it = _proteins.find(payload.path);
    if (it != _proteins.end())
    {
        Palette palette;
        for (size_t i = 0; i < payload.palette.size(); i += 3)
            palette.push_back({payload.palette[i], payload.palette[i + 1],
                               payload.palette[i + 2]});

        (*it).second->setColorScheme(payload.colorScheme, palette);
    }
    else
    {
        std::stringstream msg;
        msg << "Protein not found: " << payload.path;
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
    PLUGIN_INFO << "Selecting sequence " << payload.aminoAcidSequence
                << " on protein " << payload.path << std::endl;
    auto it = _proteins.find(payload.path);
    if (it != _proteins.end())
        (*it).second->setAminoAcidSequence(payload.aminoAcidSequence);
    else
    {
        std::stringstream msg;
        msg << "Protein not found: " << payload.path;
        PLUGIN_ERROR << msg.str() << std::endl;
        response.status = false;
        response.contents = msg.str();
    }
    return response;
}

Response Covid19Plugin::_getAminoAcidSequences(
    const AminoAcidSequencesDescriptor &payload)
{
    Response response;
    PLUGIN_INFO << "Returning sequences from protein " << payload.path
                << std::endl;
    auto it = _proteins.find(payload.path);
    if (it != _proteins.end())
    {
        const auto protein = (*it).second;
        for (const auto &sequence : protein->getSequencesAsString())
        {
            if (!response.contents.empty())
                response.contents += "\n";
            response.contents += sequence.second;
        }
        PLUGIN_INFO << response.contents << std::endl;
    }
    else
    {
        std::stringstream msg;
        msg << "Protein not found: " << payload.path;
        PLUGIN_ERROR << msg.str() << std::endl;
        response.status = false;
        response.contents = msg.str();
    }
    return response;
}

Response Covid19Plugin::_loadRNA(const RNADescriptor &payload)
{
    Response response;
    try
    {
        if (payload.range.size() != 2)
            throw std::runtime_error("Invalid range");
        const brayns::Vector2f range{payload.range[0], payload.range[1]};

        if (payload.params.size() != 3)
            throw std::runtime_error("Invalid params");
        const brayns::Vector3f params{payload.params[0], payload.params[1],
                                      payload.params[2]};

        PLUGIN_INFO << "Loading RNA sequence " << payload.name << " from "
                    << payload.path << std::endl;
        PLUGIN_INFO << "Assembly radius: " << payload.assemblyRadius
                    << std::endl;
        PLUGIN_INFO << "RNA radius     : " << payload.radius << std::endl;
        PLUGIN_INFO << "Range          : " << range << std::endl;
        PLUGIN_INFO << "Params         : " << params << std::endl;

        auto &scene = _api->getScene();
        RNASequence rnaSequence(scene, payload.name, payload.path,
                                payload.shape, payload.assemblyRadius,
                                payload.radius, range, params);
        const auto modelDescriptor = rnaSequence.getModelDescriptor();
        scene.addModel(modelDescriptor);
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
