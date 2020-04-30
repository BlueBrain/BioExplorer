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

#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include <common/types.h>

class Assembly
{
public:
    Assembly(brayns::Scene &scene,
             const AssemblyDescriptor &assemblyDescriptor);
    ~Assembly();

    void setColorScheme(const ColorSchemeDescriptor &csd);
    void setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDescriptor &aasd);
    void setAminoAcidSequenceAsRange(
        const AminoAcidSequenceAsRangeDescriptor &aasd);
    std::string getAminoAcidSequences(
        const AminoAcidSequencesDescriptor &payload) const;

    brayns::Vector4fs &getClippingPlanes() { return _clippingPlanes; }
    void setClippingPlanes(const brayns::Vector4fs &clippingPlanes)
    {
        _clippingPlanes = clippingPlanes;
    }

    void addRNASequence(const RNASequenceDescriptor &rd);
    void addProtein(const ProteinDescriptor &pd);
    void addMesh(const MeshDescriptor &md);
    void addGlycans(const GlycansDescriptor &md);

private:
    void _processInstances(brayns::ModelDescriptorPtr md,
                           const float assemblyRadius, const size_t occurrences,
                           const size_t randomSeed,
                           const brayns::Quaterniond &orientation,
                           const ModelContentType &modelType,
                           const float locationCutoffAngle = 0.f);

    brayns::Vector3f _position{0.f, 0.f, 0.f};
    bool _halfStructure{false};
    brayns::Scene &_scene;
    GlycansMap _glycans;
    ProteinMap _proteins;
    MeshMap _meshes;
    std::vector<std::pair<brayns::Vector3f, float>> _occupiedDirections;
    brayns::Vector4fs _clippingPlanes;
};

#endif // ASSEMBLY_H
