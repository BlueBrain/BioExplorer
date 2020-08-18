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

#ifndef BIOEXPLORER_PLUGIN_H
#define BIOEXPLORER_PLUGIN_H

#include <plugin/api/Params.h>

#include <brayns/pluginapi/ExtensionPlugin.h>

namespace bioexplorer
{
/**
 * @brief This class implements the BioExplorer plugin
 */
class BioExplorer : public ExtensionPlugin
{
public:
    BioExplorer();

    void init() final;
    void preRender() final;
    void postRender() final;

private:
    Response _version() const;

    // IO
    Response _exportToCache(const LoaderExportToCacheDescriptor &payload);
    Response _exportToXYZR(const LoaderExportToXYZRDescriptor &payload);

    // Movie and frames
    void _setCamera(const CameraDefinition &);
    CameraDefinition _getCamera();
    void _exportFramesToDisk(const ExportFramesToDisk &payload);
    FrameExportProgress _getFrameExportProgress();
    void _doExportFrameToDisk();

    // Biological elements
    Response _addAssembly(const AssemblyDescriptor &payload);
    Response _removeAssembly(const AssemblyDescriptor &payload);
    Response _addMembrane(const MembraneDescriptor &payload) const;
    Response _addRNASequence(const RNASequenceDescriptor &payload) const;
    Response _addProtein(const ProteinDescriptor &payload) const;
    Response _addMesh(const MeshDescriptor &payload) const;
    Response _addGlycans(const SugarsDescriptor &payload) const;
    Response _addGlucoses(const SugarsDescriptor &payload) const;

    // Other elements
    Response _addGrid(const AddGrid &payload);

    // Amino acids
    Response _setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDescriptor &payload) const;
    Response _setAminoAcidSequenceAsRange(
        const AminoAcidSequenceAsRangeDescriptor &payload) const;
    Response _getAminoAcidInformation(
        const AminoAcidInformationDescriptor &payload) const;

    // Colors and materials
    Response _setColorScheme(const ColorSchemeDescriptor &payload) const;
    Response _setMaterials(const MaterialsDescriptor &payload);
    MaterialIds _getMaterialIds(const ModelId &modelId);

    AssemblyMap _assemblies;
    bool _dirty{false};

    // Movies
    ExportFramesToDisk _exportFramesToDiskPayload;
    bool _exportFramesToDiskDirty{false};
    uint16_t _frameNumber{0};
    int16_t _accumulationFrameNumber{0};

    // Fields
    Response _visualizeFields(const VisualizeFields &payload);
    Response _exportFieldsToFile(const ExportFieldsToFile &payload);
};
} // namespace bioexplorer

#endif // BIOEXPLORER_PLUGIN_H
