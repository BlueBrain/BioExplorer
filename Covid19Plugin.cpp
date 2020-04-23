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
        PLUGIN_INFO << "Registering 'reset-assembly' endpoint" << std::endl;
        actionInterface->registerRequest<Response>("reset-assembly", [&]() {
            return _resetAssembly();
        });

        PLUGIN_INFO << "Registering 'build-assembly' endpoint" << std::endl;
        actionInterface->registerRequest<NodeDescriptor, Response>(
            "build-assembly", [&](const NodeDescriptor &payload) {
                return _buildAssembly(payload);
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

        PLUGIN_INFO << "Registering 'load-protein' endpoint" << std::endl;
        actionInterface->registerRequest<ProteinDescriptor, Response>(
            "load-protein", [&](const ProteinDescriptor &payload) {
                return _loadProtein(payload);
            });
    }
}

Response Covid19Plugin::_resetAssembly()
{
    _nodes.clear();
    _occupiedDirections.clear();
    return Response();
}

Response Covid19Plugin::_buildAssembly(const NodeDescriptor &payload)
{
    Response response;
    PLUGIN_INFO << "Initializing structure from " << payload.name << std::endl;
    PLUGIN_INFO << "Number of instances: " << payload.occurrences << std::endl;
    PLUGIN_INFO << "Assembly radius    : " << payload.assemblyRadius
                << std::endl;

    try
    {
        // Checks
        if (payload.orientation.size() != 4)
            throw std::runtime_error("Invalid orientation quaternion");

        auto &scene = _api->getScene();

        brayns::ModelDescriptorPtr modelDescriptor{nullptr};
        bool isProtein{false};
        switch (payload.modelContentType)
        {
        case ModelContentType::pdb:
        {
            ProteinPtr protein(
                new Protein(scene, {payload.name, payload.modelContents,
                                    payload.atomRadiusMultiplier}));
            modelDescriptor = protein->getModelDescriptor();
            _nodes[payload.name] = std::move(protein);
            isProtein = true;
            break;
        }
        case ModelContentType::obj:
        {
            const auto loader = brayns::MeshLoader(scene);
            uint8_ts contentAsChars;
            for (size_t i = 0; i < payload.modelContents.length(); ++i)
                contentAsChars.push_back(payload.modelContents[i]);
            brayns::Blob blob{"obj", payload.name, contentAsChars};

            modelDescriptor =
                loader.importFromBlob(std::move(blob), brayns::LoaderProgress(),
                                      brayns::PropertyMap());
            break;
        }
        default:
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

        srand(payload.randomSeed);
        size_t rnd{1};
        if (payload.randomSeed != 0 && isProtein)
            rnd = rand() % payload.occurrences;

        size_t instanceCount = 0;
        for (size_t i = 0; i < payload.occurrences; ++i)
        {
            // Randomizer
            float assemblyRadius = payload.assemblyRadius;
            if (payload.randomSeed != 0 && !isProtein)
                assemblyRadius *= 1.f + (float(rand() % 20) / 1000.f);

            // Sphere filling
            const float y = ((i * offset) - 1.f) + (offset / 2.f);
            const float r = sqrt(1.f - pow(y, 2.f));
            const float phi = ((i + rnd) % payload.occurrences) * increment;
            const float x = cos(phi) * r;
            const float z = sin(phi) * r;
            const brayns::Vector3f direction{x, y, z};

            // Remove membrane where proteins are. This is currently done
            // according to the vector orientation
            bool occupied{false};
            if (!isProtein)
                for (const auto &occupiedDirection : _occupiedDirections)
                    if (dot(direction, occupiedDirection.first) >
                        occupiedDirection.second)
                    {
                        occupied = true;
                        break;
                    }
            if (occupied)
                continue;

            // Half structure
            if (payload.halfStructure &&
                (direction.x > 0.f && direction.y > 0.f && direction.z > 0.f))
                continue;

            // Final transformation
            brayns::Transformation tf;
            const brayns::Vector3f position = assemblyRadius * direction;
            tf.setTranslation(position - center);
            tf.setRotationCenter(center);

            const brayns::Quaterniond proteinOrientation = {
                payload.orientation[0], payload.orientation[1],
                payload.orientation[2], payload.orientation[3]};

            brayns::Quaterniond assemblyOrientation =
                glm::quatLookAt(direction, {0.f, 1.f, 0.f});

            tf.setRotation(assemblyOrientation * proteinOrientation);

            if (instanceCount == 0)
                modelDescriptor->setTransformation(tf);
            else
            {
                brayns::ModelInstance instance(true, false, tf);
                modelDescriptor->addInstance(instance);
            }
            ++instanceCount;

            // Store occupied direction
            if (isProtein)
                _occupiedDirections.push_back(
                    {direction, payload.locationCutoffAngle});
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
    auto it = _nodes.find(payload.name);
    if (it != _nodes.end())
    {
        Protein *node = dynamic_cast<Protein *>((*it).second.get());
        if (node)
        {
            Palette palette;
            for (size_t i = 0; i < payload.palette.size(); i += 3)
                palette.push_back({payload.palette[i], payload.palette[i + 1],
                                   payload.palette[i + 2]});

            node->setColorScheme(payload.colorScheme, palette);
        }
    }
    else
    {
        std::stringstream msg;
        msg << "Protein not found: " << payload.name;
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
                << " on protein " << payload.name << std::endl;
    auto it = _nodes.find(payload.name);
    if (it != _nodes.end())
    {
        auto node = dynamic_cast<Protein *>((*it).second.get());
        if (node)
            node->setAminoAcidSequence(payload.aminoAcidSequence);
    }
    else
    {
        std::stringstream msg;
        msg << "Protein not found: " << payload.name;
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
    PLUGIN_INFO << "Returning sequences from protein " << payload.name
                << std::endl;
    auto it = _nodes.find(payload.name);
    if (it != _nodes.end())
    {
        auto node = dynamic_cast<Protein *>((*it).second.get());
        if (node)
        {
            for (const auto &sequence : node->getSequencesAsString())
            {
                if (!response.contents.empty())
                    response.contents += "\n";
                response.contents += sequence.second;
            }
            PLUGIN_INFO << response.contents << std::endl;
        }
    }
    else
    {
        std::stringstream msg;
        msg << "Protein not found: " << payload.name;
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

Response Covid19Plugin::_loadProtein(const ProteinDescriptor &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO << "Loading Protein " << payload.name << std::endl;
        PLUGIN_INFO << "Radius multiplier: " << payload.atomRadiusMultiplier
                    << std::endl;

        auto &scene = _api->getScene();

        ProteinPtr protein(new Protein(scene, payload));
        const auto modelDescriptor = protein->getModelDescriptor();
        scene.addModel(modelDescriptor);
        _nodes[payload.name] = std::move(protein);
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
