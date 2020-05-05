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

#include <api/BioExplorerParams.h>

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

private:
    Response _version() const;
    Response _addAssembly(const AssemblyDescriptor &payload);
    Response _removeAssembly(const AssemblyDescriptor &payload);
    Response _addRNASequence(const RNASequenceDescriptor &payload) const;
    Response _addProtein(const ProteinDescriptor &payload) const;
    Response _addMesh(const MeshDescriptor &payload) const;
    Response _addGlycans(const GlycansDescriptor &payload) const;

    Response _setColorScheme(const ColorSchemeDescriptor &payload) const;
    Response _setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDescriptor &payload) const;
    Response _setAminoAcidSequenceAsRange(
        const AminoAcidSequenceAsRangeDescriptor &payload) const;
    Response _getAminoAcidSequences(
        const AminoAcidSequencesDescriptor &payload) const;

    AssemblyMap _assemblies;
};
} // namespace bioexplorer

#endif // BIOEXPLORER_PLUGIN_H
