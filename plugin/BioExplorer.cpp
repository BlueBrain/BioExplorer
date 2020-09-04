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

#include <plugin/bioexplorer/Assembly.h>
#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>
#include <plugin/io/BioExplorerLoader.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/FrameBuffer.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

#include <fstream>

namespace bioexplorer
{
void _addBioExplorerRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO << "Registering 'bio_explorer' renderer" << std::endl;
    brayns::PropertyMap properties;
    properties.setProperty(
        {"giDistance", 10000., {"Global illumination distance"}});
    properties.setProperty(
        {"giWeight", 0., 1., 1., {"Global illumination weight"}});
    properties.setProperty(
        {"giSamples", 0, 0, 64, {"Global illumination samples"}});
    properties.setProperty({"shadows", 0., 0., 1., {"Shadow intensity"}});
    properties.setProperty({"softShadows", 0., 0., 1., {"Shadow softness"}});
    properties.setProperty(
        {"softShadowsSamples", 1, 1, 64, {"Soft shadow samples"}});
    properties.setProperty({"exposure", 1., 0.01, 10., {"Exposure"}});
    properties.setProperty({"fogStart", 0., 0., 1e6, {"Fog start"}});
    properties.setProperty({"fogThickness", 1e6, 1e6, 1e6, {"Fog thickness"}});
    properties.setProperty(
        {"maxBounces", 3, 1, 100, {"Maximum number of ray bounces"}});
    properties.setProperty({"useHardwareRandomizer",
                            false,
                            {"Use hardware accelerated randomizer"}});
    properties.setProperty({"showBackground", false, {"Show background"}});
    engine.addRendererType("bio_explorer", properties);
}

void _addBioExplorerFieldsRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO << "Registering 'bio_explorer_fields' renderer" << std::endl;
    brayns::PropertyMap properties;
    properties.setProperty({"exposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"useHardwareRandomizer",
                            false,
                            {"Use hardware accelerated randomizer"}});
    properties.setProperty({"step", 0.1, 0.001, 10.0, {"Ray step"}});
    properties.setProperty(
        {"maxSteps", 64, 32, 2048, {"Maximum number of ray steps"}});
    properties.setProperty({"cutoff", 250.0, 0.0, 2000.0, {"cutoff"}});
    engine.addRendererType("bio_explorer_fields", properties);
}

void _addBioExplorerPerspectiveCamera(brayns::Engine &engine)
{
    PLUGIN_INFO << "Registering 'bio_explorer_perspective' camera" << std::endl;

    brayns::PropertyMap properties;
    properties.setProperty({"fovy", 45., .1, 360., {"Field of view"}});
    properties.setProperty({"aspect", 1., {"Aspect ratio"}});
    properties.setProperty({"apertureRadius", 0., {"Aperture radius"}});
    properties.setProperty({"focusDistance", 1., {"Focus Distance"}});
    properties.setProperty({"enableClippingPlanes", true, {"Clipping"}});
    properties.setProperty({"stereo", false, {"Stereo"}});
    properties.setProperty(
        {"interpupillaryDistance", 0.0635, 0.0, 10.0, {"Eye separation"}});
    engine.addCameraType("bio_explorer_perspective", properties);
}

BioExplorer::BioExplorer()
    : ExtensionPlugin()
{
}

void BioExplorer::init()
{
    auto actionInterface = _api->getActionInterface();
    auto &scene = _api->getScene();
    auto &registry = scene.getLoaderRegistry();

    registry.registerLoader(std::make_unique<BioExplorerLoader>(
        scene, BioExplorerLoader::getCLIProperties()));

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

        PLUGIN_INFO
            << "Registering 'get-protein-amino-acid-information' endpoint"
            << std::endl;
        actionInterface
            ->registerRequest<AminoAcidInformationDescriptor, Response>(
                "get-protein-amino-acid-information",
                [&](const AminoAcidInformationDescriptor &payload) {
                    return _getAminoAcidInformation(payload);
                });

        PLUGIN_INFO << "Registering 'add-rna-sequence' endpoint" << std::endl;
        actionInterface->registerRequest<RNASequenceDescriptor, Response>(
            "add-rna-sequence", [&](const RNASequenceDescriptor &payload) {
                return _addRNASequence(payload);
            });

        PLUGIN_INFO << "Registering 'add-membrane' endpoint" << std::endl;
        actionInterface->registerRequest<MembraneDescriptor, Response>(
            "add-membrane", [&](const MembraneDescriptor &payload) {
                return _addMembrane(payload);
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
        actionInterface->registerRequest<SugarsDescriptor, Response>(
            "add-glycans", [&](const SugarsDescriptor &payload) {
                return _addGlycans(payload);
            });

        PLUGIN_INFO << "Registering 'add-sugars' endpoint" << std::endl;
        actionInterface->registerRequest<SugarsDescriptor, Response>(
            "add-sugars", [&](const SugarsDescriptor &payload) {
                return _addSugars(payload);
            });

        PLUGIN_INFO << "Registering 'export-to-cache' endpoint" << std::endl;
        actionInterface->registerRequest<FileAccess, Response>(
            "export-to-cache",
            [&](const FileAccess &payload) { return _exportToCache(payload); });

        PLUGIN_INFO << "Registering 'export-to-xyzr' endpoint" << std::endl;
        actionInterface->registerRequest<FileAccess, Response>(
            "export-to-xyzr",
            [&](const FileAccess &payload) { return _exportToXYZR(payload); });

        PLUGIN_INFO << "Registering 'add-grid' endpoint" << std::endl;
        actionInterface->registerNotification<AddGrid>(
            "add-grid", [&](const AddGrid &payload) { _addGrid(payload); });

        PLUGIN_INFO << "Registering 'set-materials' endpoint" << std::endl;
        actionInterface->registerNotification<MaterialsDescriptor>(
            "set-materials",
            [&](const MaterialsDescriptor &param) { _setMaterials(param); });

        PLUGIN_INFO << "Registering 'get-material-ids' endpoint" << std::endl;
        actionInterface->registerRequest<ModelId, MaterialIds>(
            "get-material-ids", [&](const ModelId &modelId) -> MaterialIds {
                return _getMaterialIds(modelId);
            });

        PLUGIN_INFO << "Registering 'set-odu-camera' endpoint" << std::endl;
        actionInterface->registerNotification<CameraDefinition>(
            "set-odu-camera",
            [&](const CameraDefinition &s) { _setCamera(s); });

        PLUGIN_INFO << "Registering 'get-odu-camera' endpoint" << std::endl;
        actionInterface->registerRequest<CameraDefinition>(
            "get-odu-camera",
            [&]() -> CameraDefinition { return _getCamera(); });

        PLUGIN_INFO << "Registering 'export-frames-to-disk' endpoint"
                    << std::endl;
        actionInterface->registerNotification<ExportFramesToDisk>(
            "export-frames-to-disk",
            [&](const ExportFramesToDisk &s) { _exportFramesToDisk(s); });

        PLUGIN_INFO << "Registering 'get-export-frames-progress' endpoint"
                    << std::endl;
        actionInterface->registerRequest<FrameExportProgress>(
            "get-export-frames-progress", [&](void) -> FrameExportProgress {
                return _getFrameExportProgress();
            });

        PLUGIN_INFO << "Registering 'visualize-fields' endpoint" << std::endl;
        actionInterface->registerRequest<BuildFields, Response>(
            "build-fields",
            [&](const BuildFields &s) { return _buildFields(s); });

        PLUGIN_INFO << "Registering 'export-fields-to-file' endpoint"
                    << std::endl;
        actionInterface->registerRequest<ModelIdFileAccess, Response>(
            "export-fields-to-file",
            [&](const ModelIdFileAccess &s) { return _exportFieldsToFile(s); });

        PLUGIN_INFO << "Registering 'import-fields-from-file' endpoint"
                    << std::endl;
        actionInterface->registerRequest<FileAccess, Response>(
            "import-fields-from-file",
            [&](const FileAccess &s) { return _importFieldsFromFile(s); });
    }

    auto &engine = _api->getEngine();
    _addBioExplorerPerspectiveCamera(engine);
    _addBioExplorerRenderer(engine);
    _addBioExplorerFieldsRenderer(engine);
}

void BioExplorer::preRender()
{
    if (_dirty)
        _api->getScene().markModified();
    _dirty = false;

    if (_exportFramesToDiskDirty && _accumulationFrameNumber == 0)
    {
        const auto &ai = _exportFramesToDiskPayload.animationInformation;
        if (_frameNumber >= ai.size())
            _exportFramesToDiskDirty = false;
        else
        {
            const uint64_t i = 12 * _frameNumber;
            // Camera position
            CameraDefinition cd;
            const auto &ci = _exportFramesToDiskPayload.cameraInformation;
            cd.origin = {ci[i], ci[i + 1], ci[i + 2]};
            cd.direction = {ci[i + 3], ci[i + 4], ci[i + 5]};
            cd.up = {ci[i + 6], ci[i + 7], ci[i + 8]};
            cd.apertureRadius = ci[i + 9];
            cd.focusDistance = ci[i + 10];
            cd.interpupillaryDistance = ci[i + 11];
            _setCamera(cd);

            // Animation parameters
            _api->getParametersManager().getAnimationParameters().setFrame(
                ai[_frameNumber]);
        }
    }
}

void BioExplorer::postRender()
{
    if (_exportFramesToDiskDirty &&
        _accumulationFrameNumber == _exportFramesToDiskPayload.spp)
    {
        _doExportFrameToDisk();
        ++_frameNumber;
        _accumulationFrameNumber = 0;
    }
    else
        ++_accumulationFrameNumber;
}

Response BioExplorer::_version() const
{
    Response response;
    response.contents = PLUGIN_VERSION;
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
        AssemblyPtr assembly = AssemblyPtr(new Assembly(scene, payload));
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

Response BioExplorer::_setColorScheme(
    const ColorSchemeDescriptor &payload) const
{
    Response response;
    try
    {
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
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
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

Response BioExplorer::_getAminoAcidInformation(
    const AminoAcidInformationDescriptor &payload) const
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            response.contents = (*it).second->getAminoAcidInformation(payload);
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

Response BioExplorer::_addMembrane(const MembraneDescriptor &payload) const
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addMembrane(payload);
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
    _api->getEngine().triggerRender();
    return response;
}

Response BioExplorer::_addGlycans(const SugarsDescriptor &payload) const
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

Response BioExplorer::_addSugars(const SugarsDescriptor &payload) const
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addSugars(payload);
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

Response BioExplorer::_exportToCache(const FileAccess &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        BioExplorerLoader loader(scene);
        loader.exportToCache(payload.filename);
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response BioExplorer::_exportToXYZR(const FileAccess &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        BioExplorerLoader loader(scene);
        loader.exportToXYZR(payload.filename);
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

Response BioExplorer::_addGrid(const AddGrid &payload)
{
    Response response;
    try
    {
        BRAYNS_INFO << "Building Grid scene" << std::endl;

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const brayns::Vector3f red = {1, 0, 0};
        const brayns::Vector3f green = {0, 1, 0};
        const brayns::Vector3f blue = {0, 0, 1};
        const brayns::Vector3f grey = {0.5, 0.5, 0.5};

        brayns::PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});

        auto material = model->createMaterial(0, "x");
        material->setDiffuseColor(grey);
        material->setProperties(props);
        const auto &position =
            brayns::Vector3f(payload.position[0], payload.position[1],
                             payload.position[2]);

        const float m = payload.minValue;
        const float M = payload.maxValue;
        const float s = payload.steps;
        const float r = payload.radius;
        for (float x = m; x <= M; x += s)
            for (float y = m; y <= M; y += s)
                if (fabs(x) < 0.001f || fabs(y) < 0.001f)
                {
                    model->addCylinder(0, {position + brayns::Vector3f(x, y, m),
                                           position + brayns::Vector3f(x, y, M),
                                           r});
                    model->addCylinder(0, {position + brayns::Vector3f(m, x, y),
                                           position + brayns::Vector3f(M, x, y),
                                           r});
                    model->addCylinder(0, {position + brayns::Vector3f(x, m, y),
                                           position + brayns::Vector3f(x, M, y),
                                           r});
                }

        material = model->createMaterial(1, "plane_x");
        material->setDiffuseColor(payload.useColors ? red : grey);
        material->setOpacity(payload.planeOpacity);
        material->setProperties(props);
        auto &tmx = model->getTriangleMeshes()[1];
        tmx.vertices.push_back(position + brayns::Vector3f(m, 0, m));
        tmx.vertices.push_back(position + brayns::Vector3f(M, 0, m));
        tmx.vertices.push_back(position + brayns::Vector3f(M, 0, M));
        tmx.vertices.push_back(position + brayns::Vector3f(m, 0, M));
        tmx.indices.push_back(brayns::Vector3ui(0, 1, 2));
        tmx.indices.push_back(brayns::Vector3ui(2, 3, 0));

        material = model->createMaterial(2, "plane_y");
        material->setDiffuseColor(payload.useColors ? green : grey);
        material->setOpacity(payload.planeOpacity);
        material->setProperties(props);
        auto &tmy = model->getTriangleMeshes()[2];
        tmy.vertices.push_back(position + brayns::Vector3f(m, m, 0));
        tmy.vertices.push_back(position + brayns::Vector3f(M, m, 0));
        tmy.vertices.push_back(position + brayns::Vector3f(M, M, 0));
        tmy.vertices.push_back(position + brayns::Vector3f(m, M, 0));
        tmy.indices.push_back(brayns::Vector3ui(0, 1, 2));
        tmy.indices.push_back(brayns::Vector3ui(2, 3, 0));

        material = model->createMaterial(3, "plane_z");
        material->setDiffuseColor(payload.useColors ? blue : grey);
        material->setOpacity(payload.planeOpacity);
        material->setProperties(props);
        auto &tmz = model->getTriangleMeshes()[3];
        tmz.vertices.push_back(position + brayns::Vector3f(0, m, m));
        tmz.vertices.push_back(position + brayns::Vector3f(0, m, M));
        tmz.vertices.push_back(position + brayns::Vector3f(0, M, M));
        tmz.vertices.push_back(position + brayns::Vector3f(0, M, m));
        tmz.indices.push_back(brayns::Vector3ui(0, 1, 2));
        tmz.indices.push_back(brayns::Vector3ui(2, 3, 0));

        if (payload.showAxis)
        {
            const float l = M;
            const float smallRadius = payload.radius * 25.0;
            const float largeRadius = payload.radius * 50.0;
            const float l1 = l * 0.89;
            const float l2 = l * 0.90;

            brayns::PropertyMap props;
            props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
            props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                               static_cast<int>(MaterialShadingMode::basic)});

            // X
            material = model->createMaterial(4, "x_axis");
            material->setDiffuseColor({1, 0, 0});
            material->setProperties(props);

            model->addCylinder(4,
                               {position, position + brayns::Vector3f(l1, 0, 0),
                                smallRadius});
            model->addCone(4, {position + brayns::Vector3f(l1, 0, 0),
                               position + brayns::Vector3f(l2, 0, 0),
                               smallRadius, largeRadius});
            model->addCone(4, {position + brayns::Vector3f(l2, 0, 0),
                               position + brayns::Vector3f(M, 0, 0),
                               largeRadius, 0});

            // Y
            material = model->createMaterial(5, "y_axis");
            material->setDiffuseColor({0, 1, 0});
            material->setProperties(props);

            model->addCylinder(5,
                               {position, position + brayns::Vector3f(0, l1, 0),
                                smallRadius});
            model->addCone(5, {position + brayns::Vector3f(0, l1, 0),
                               position + brayns::Vector3f(0, l2, 0),
                               smallRadius, largeRadius});
            model->addCone(5, {position + brayns::Vector3f(0, l2, 0),
                               position + brayns::Vector3f(0, M, 0),
                               largeRadius, 0});

            // Z
            material = model->createMaterial(6, "z_axis");
            material->setDiffuseColor({0, 0, 1});
            material->setProperties(props);

            model->addCylinder(6,
                               {position, position + brayns::Vector3f(0, 0, l1),
                                smallRadius});
            model->addCone(6, {position + brayns::Vector3f(0, 0, l1),
                               position + brayns::Vector3f(0, 0, l2),
                               smallRadius, largeRadius});
            model->addCone(6, {position + brayns::Vector3f(0, 0, l2),
                               position + brayns::Vector3f(0, 0, M),
                               largeRadius, 0});

            // Origin
            model->addSphere(0, {position, smallRadius});
        }

        scene.addModel(
            std::make_shared<brayns::ModelDescriptor>(std::move(model),
                                                      "Grid"));
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

MaterialIds BioExplorer::_getMaterialIds(const ModelId &modelId)
{
    MaterialIds materialIds;
    auto modelDescriptor = _api->getScene().getModel(modelId.modelId);
    if (modelDescriptor)
    {
        for (const auto &material : modelDescriptor->getModel().getMaterials())
            if (material.first != brayns::BOUNDINGBOX_MATERIAL_ID &&
                material.first != brayns::SECONDARY_MODEL_MATERIAL_ID)
                materialIds.ids.push_back(material.first);
    }
    else
        PLUGIN_THROW(std::runtime_error("Invalid model ID"));
    return materialIds;
}

Response BioExplorer::_setMaterials(const MaterialsDescriptor &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        for (const auto modelId : payload.modelIds)
        {
            PLUGIN_INFO << "Modifying materials on model " << modelId
                        << std::endl;
            auto modelDescriptor = scene.getModel(modelId);
            if (modelDescriptor)
            {
                size_t id = 0;
                for (const auto materialId : payload.materialIds)
                {
                    try
                    {
                        auto material =
                            modelDescriptor->getModel().getMaterial(materialId);
                        if (material)
                        {
                            if (!payload.diffuseColors.empty())
                            {
                                const size_t index = id * 3;
                                material->setDiffuseColor(
                                    {payload.diffuseColors[index],
                                     payload.diffuseColors[index + 1],
                                     payload.diffuseColors[index + 2]});
                                material->setSpecularColor(
                                    {payload.specularColors[index],
                                     payload.specularColors[index + 1],
                                     payload.specularColors[index + 2]});
                            }

                            if (!payload.specularExponents.empty())
                                material->setSpecularExponent(
                                    payload.specularExponents[id]);
                            if (!payload.reflectionIndices.empty())
                                material->setReflectionIndex(
                                    payload.reflectionIndices[id]);
                            if (!payload.opacities.empty())
                                material->setOpacity(payload.opacities[id]);
                            if (!payload.refractionIndices.empty())
                                material->setRefractionIndex(
                                    payload.refractionIndices[id]);
                            if (!payload.emissions.empty())
                                material->setEmission(payload.emissions[id]);
                            if (!payload.glossinesses.empty())
                                material->setGlossiness(
                                    payload.glossinesses[id]);
                            if (!payload.shadingModes.empty())
                                material->updateProperty(
                                    MATERIAL_PROPERTY_SHADING_MODE,
                                    payload.shadingModes[id]);
                            if (!payload.userParameters.empty())
                                material->updateProperty(
                                    MATERIAL_PROPERTY_USER_PARAMETER,
                                    static_cast<double>(
                                        payload.userParameters[id]));

                            // This is needed to apply modifications. Changes to
                            // the material will be committed after the
                            // rendering of the current frame is completed
                            material->markModified();
                        }
                    }
                    catch (const std::runtime_error &e)
                    {
                        PLUGIN_INFO << e.what() << std::endl;
                    }
                    ++id;
                }
                _dirty = true;
            }
            else
                PLUGIN_INFO << "Model " << modelId << " is not registered"
                            << std::endl;
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

void BioExplorer::_setCamera(const CameraDefinition &payload)
{
    auto &camera = _api->getCamera();

    // Origin
    const auto &o = payload.origin;
    brayns::Vector3f origin{o[0], o[1], o[2]};
    camera.setPosition(origin);

    // Target
    const auto &d = payload.direction;
    brayns::Vector3f direction{d[0], d[1], d[2]};
    camera.setTarget(origin + direction);

    // Up
    const auto &u = payload.up;
    brayns::Vector3f up{u[0], u[1], u[2]};

    // Orientation
    const glm::quat q = glm::inverse(
        glm::lookAt(origin, origin + direction,
                    up)); // Not quite sure why this should be inverted?!?
    camera.setOrientation(q);

    // Aperture
    camera.updateProperty("apertureRadius", payload.apertureRadius);

    // Focus distance
    camera.updateProperty("focusDistance", payload.focusDistance);

    // Stereo
    camera.updateProperty("stereo", payload.focusDistance != 0.0);
    camera.updateProperty("interpupillaryDistance",
                          payload.interpupillaryDistance);

    _api->getCamera().markModified();

    PLUGIN_DEBUG << "SET: " << origin << ", " << direction << ", " << up << ", "
                 << glm::inverse(q) << "," << payload.apertureRadius << ","
                 << payload.focusDistance << std::endl;
}

CameraDefinition BioExplorer::_getCamera()
{
    const auto &camera = _api->getCamera();

    CameraDefinition cd;
    const auto &p = camera.getPosition();
    cd.origin = {p.x, p.y, p.z};
    const auto d =
        glm::rotate(camera.getOrientation(), brayns::Vector3d(0., 0., -1.));
    cd.direction = {d.x, d.y, d.z};
    const auto u =
        glm::rotate(camera.getOrientation(), brayns::Vector3d(0., 1., 0.));
    cd.up = {u.x, u.y, u.z};
    PLUGIN_DEBUG << "GET: " << p << ", " << d << ", " << u << ", "
                 << camera.getOrientation() << std::endl;
    return cd;
}

void BioExplorer::_exportFramesToDisk(const ExportFramesToDisk &payload)
{
    _exportFramesToDiskPayload = payload;
    _exportFramesToDiskDirty = true;
    _frameNumber = payload.startFrame;
    _accumulationFrameNumber = 0;
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    frameBuffer.clear();
    PLUGIN_INFO << "-----------------------------------------------------------"
                   "---------------------"
                << std::endl;
    PLUGIN_INFO << "Movie settings     :" << std::endl;
    PLUGIN_INFO << "- Number of frames : "
                << payload.animationInformation.size() - payload.startFrame
                << std::endl;
    PLUGIN_INFO << "- Samples per pixel: " << payload.spp << std::endl;
    PLUGIN_INFO << "- Frame size       : " << frameBuffer.getSize()
                << std::endl;
    PLUGIN_INFO << "- Export folder    : " << payload.path << std::endl;
    PLUGIN_INFO << "- Start frame      : " << payload.startFrame << std::endl;
    PLUGIN_INFO << "-----------------------------------------------------------"
                   "---------------------"
                << std::endl;
}

void BioExplorer::_doExportFrameToDisk()
{
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    auto image = frameBuffer.getImage();
    auto fif = _exportFramesToDiskPayload.format == "jpg"
                   ? FIF_JPEG
                   : FreeImage_GetFIFFromFormat(
                         _exportFramesToDiskPayload.format.c_str());
    if (fif == FIF_JPEG)
        image.reset(FreeImage_ConvertTo24Bits(image.get()));
    else if (fif == FIF_UNKNOWN)
        throw std::runtime_error("Unknown format: " +
                                 _exportFramesToDiskPayload.format);

    int flags = _exportFramesToDiskPayload.quality;
    if (fif == FIF_TIFF)
        flags = TIFF_NONE;

    brayns::freeimage::MemoryPtr memory(FreeImage_OpenMemory());

    FreeImage_SaveToMemory(fif, image.get(), memory.get(), flags);

    BYTE *pixels = nullptr;
    DWORD numPixels = 0;
    FreeImage_AcquireMemory(memory.get(), &pixels, &numPixels);

    char frame[7];
    sprintf(frame, "%05d", _frameNumber);
    std::string filename = _exportFramesToDiskPayload.path + '/' + frame + "." +
                           _exportFramesToDiskPayload.format;
    std::ofstream file;
    file.open(filename, std::ios_base::binary);
    if (!file.is_open())
        PLUGIN_THROW(std::runtime_error("Failed to create " + filename));

    file.write((char *)pixels, numPixels);
    file.close();

    frameBuffer.clear();

    PLUGIN_INFO << "Frame saved to " << filename << std::endl;
}

FrameExportProgress BioExplorer::_getFrameExportProgress()
{
    FrameExportProgress result;
    const size_t totalNumberOfFrames =
        (_exportFramesToDiskPayload.animationInformation.size() -
         _exportFramesToDiskPayload.startFrame) *
        _exportFramesToDiskPayload.spp;
    const float currentProgress =
        _frameNumber * _exportFramesToDiskPayload.spp +
        _accumulationFrameNumber;

    result.progress = currentProgress / float(totalNumberOfFrames);
    return result;
}

void BioExplorer::_attachFieldsHandler(FieldsHandlerPtr handler)
{
    auto &scene = _api->getScene();
    auto model = scene.createModel();
    const auto &spacing = Vector3f(handler->getSpacing());
    const auto &size = Vector3f(handler->getDimensions()) * spacing;
    const auto &offset = Vector3f(handler->getOffset());
    const brayns::Vector3f center{(offset + size / 2.f)};

    auto material = model->createMaterial(0, "default");
    model->addSphere(0, {center,
                         std::max(std::max(size.x, size.y), size.z) / 2.f});
    ModelMetadata metadata;
    metadata["Center"] = std::to_string(center.x) + "," +
                         std::to_string(center.y) + "," +
                         std::to_string(center.z);
    metadata["Size"] = std::to_string(size.x) + "," + std::to_string(size.y) +
                       "," + std::to_string(size.z);
    metadata["Spacing"] = std::to_string(spacing.x) + "," +
                          std::to_string(spacing.y) + "," +
                          std::to_string(spacing.z);

    model->setSimulationHandler(handler);
    setTransferFunction(model->getTransferFunction());

    auto modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), "Fields", metadata);
    scene.addModel(modelDescriptor);
}

Response BioExplorer::_buildFields(const BuildFields &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO << "Building Fields from scene" << std::endl;
        auto &scene = _api->getScene();
        FieldsHandlerPtr handler =
            std::make_shared<FieldsHandler>(scene, payload.voxelSize);
        _attachFieldsHandler(handler);
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response BioExplorer::_importFieldsFromFile(const FileAccess &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO << "Importing Fields from " << payload.filename
                    << std::endl;
        FieldsHandlerPtr handler =
            std::make_shared<FieldsHandler>(payload.filename);
        _attachFieldsHandler(handler);
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response BioExplorer::_exportFieldsToFile(const ModelIdFileAccess &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO << "Exporting fields to " << payload.filename << std::endl;
        const auto &scene = _api->getScene();
        auto modelDescriptor = scene.getModel(payload.modelId);
        if (modelDescriptor)
        {
            auto handler = modelDescriptor->getModel().getSimulationHandler();
            if (handler)
            {
                FieldsHandler *fieldsHandler =
                    dynamic_cast<FieldsHandler *>(handler.get());
                fieldsHandler->exportToFile(payload.filename);
            }
        }
        else
            PLUGIN_THROW(std::runtime_error("Unknown model ID"));
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
    PLUGIN_INFO << " _|_|_|    _|            _|_|_|_|                      _|  "
                   "                                      "
                << std::endl;
    PLUGIN_INFO << " _|    _|        _|_|    _|        _|    _|  _|_|_|    _|  "
                   "  _|_|    _|  _|_|    _|_|    _|  _|_|"
                << std::endl;
    PLUGIN_INFO << " _|_|_|    _|  _|    _|  _|_|_|      _|_|    _|    _|  _|  "
                   "_|    _|  _|_|      _|_|_|_|  _|_|    "
                << std::endl;
    PLUGIN_INFO << " _|    _|  _|  _|    _|  _|        _|    _|  _|    _|  _|  "
                   "_|    _|  _|        _|        _|      "
                << std::endl;
    PLUGIN_INFO << " _|_|_|    _|    _|_|    _|_|_|_|  _|    _|  _|_|_|    _|  "
                   "  _|_|    _|          _|_|_|  _|      "
                << std::endl;
    PLUGIN_INFO << "                                             _|            "
                   "                                      "
                << std::endl;
    PLUGIN_INFO << "                                             _|            "
                   "                                      "
                << std::endl;
    PLUGIN_INFO << "Initializing BioExplorer plug-in (version "
                << PLUGIN_VERSION << ")" << std::endl;
    PLUGIN_INFO << std::endl;
    return new BioExplorer();
}

} // namespace bioexplorer
