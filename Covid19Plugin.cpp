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
#include "log.h"

#include <brayns/common/ActionInterface.h>
#include <brayns/common/Progress.h>
#include <brayns/common/utils/enumUtils.h>
#include <brayns/common/utils/utils.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/io/MeshLoader.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

#include <io/ProteinLoader.h>

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
        actionInterface->registerNotification<StructureDescriptor>(
            "build-structure", [&](const StructureDescriptor &payload) {
                _buildStructure(payload);
            });
    }
}

void Covid19Plugin::preRender() {}

void Covid19Plugin::_buildStructure(const StructureDescriptor &payload)
{
    PLUGIN_INFO << "Initializing structure from " << payload.filename
                << std::endl;
    PLUGIN_INFO << "Number of instances: " << payload.instances << std::endl;
    PLUGIN_INFO << "Virus radius    : " << payload.assemblyRadius << std::endl;
    //    PLUGIN_INFO << "Color scheme    : "
    //                << brayns::enumToString(payload.colorScheme) << std::endl;

    auto &scene = _api->getScene();
    const std::string ext = brayns::extractExtension(payload.filename);

    brayns::ModelDescriptorPtr modelDescriptor = nullptr;
    brayns::PropertyMap props;
    if (ext == "pdb" || ext == "pdb1")
    {
        auto loader = ProteinLoader(scene);
        const LoaderParameters params = {payload.atomRadiusMultiplier,
                                         payload.colorScheme,
                                         payload.transmembraneSequence};
        modelDescriptor = loader.importFromFile(payload.filename, params);
        scene.addModel(modelDescriptor);
    }
    else if (ext == "obj")
    {
        const auto loader = brayns::MeshLoader(scene);
        modelDescriptor =
            loader.importFromFile(payload.filename, brayns::LoaderProgress(),
                                  brayns::PropertyMap());
        scene.addModel(modelDescriptor);
    }
    else
        PLUGIN_THROW(std::runtime_error("Unsupported file format"));

    const auto &model = modelDescriptor->getModel();
    const auto &bounds = model.getBounds();
    const brayns::Vector3f &center = bounds.getCenter();

    const float offset = 2.f / payload.instances;
    const float increment = M_PI * (3.f - sqrt(5.f));
    srand(time(NULL));
    size_t rnd = payload.randomize ? rand() % payload.instances : 1;

    size_t instanceCount = 0;
    for (size_t i = 0; i < payload.instances; ++i)
    {
#if 0
        const float y = ((i * offset) - 1.f) + (offset / 2.f);
        const float r = sqrt(1.f - pow(y, 2.f));
        const float phi = ((i + rnd) % payload.instances) * increment;
        const float x = cos(phi) * r;
        const float z = sin(phi) * r;
        const auto direction = brayns::Vector3f(x, y, z);
#else
        const float angle = payload.instances / 2.f * M_PI;
        const float phi = acos(1 - 2 * (i + 0.5) / payload.instances);
        const float theta = M_PI * (1 + pow(5, 0.5)) * i;
        const float x =
            payload.deformation[0] * cos(angle * i) + cos(theta) * sin(phi);
        const float y =
            payload.deformation[1] * sin(angle * i) + sin(theta) * sin(phi);
        const float z = cos(phi);
        const auto direction = brayns::Vector3f(x, y, z);

#endif

        if (payload.halfStructure &&
            (direction.x > 0.f && direction.y > 0.f && direction.z > 0.f))
            continue;

        brayns::Transformation tf;
        const brayns::Vector3f position = payload.assemblyRadius * direction;
        tf.setTranslation(position - center);
        tf.setRotationCenter(center);

        const brayns::Vector3f up = {0.f, 1.f, 0.f};
        const brayns::Quaterniond quat = glm::quatLookAt(direction, up);
        tf.setRotation(quat);

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

extern "C" brayns::ExtensionPlugin *brayns_plugin_create(int /*argc*/,
                                                         char ** /*argv*/)
{
    PLUGIN_INFO << "Initializing Covid19 plugin" << std::endl;
    return new Covid19Plugin();
}
