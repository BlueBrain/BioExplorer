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

#ifndef BIOEXPLORER_ASSEMBLY_H
#define BIOEXPLORER_ASSEMBLY_H

#include <common/types.h>

namespace bioexplorer
{
class Assembly
{
public:
    Assembly(Scene &scene, const AssemblyDescriptor &assemblyDescriptor);
    ~Assembly();

    void setColorScheme(const ColorSchemeDescriptor &csd);
    void setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDescriptor &aasd);
    void setAminoAcidSequenceAsRange(
        const AminoAcidSequenceAsRangeDescriptor &aasd);
    std::string getAminoAcidSequences(
        const AminoAcidSequencesDescriptor &payload) const;

    Vector4fs &getClippingPlanes() { return _clippingPlanes; }
    void setClippingPlanes(const Vector4fs &clippingPlanes)
    {
        _clippingPlanes = clippingPlanes;
    }

    void addRNASequence(const RNASequenceDescriptor &rd);
    void addProtein(const ProteinDescriptor &pd);
    void addMesh(const MeshDescriptor &md);
    void addGlycans(const GlycansDescriptor &md);

private:
    void _processInstances(ModelDescriptorPtr md, const float assemblyRadius,
                           const size_t occurrences, const size_t randomSeed,
                           const Quaterniond &orientation,
                           const PositionRandomizationType &randomizationType,
                           const float locationCutoffAngle = 0.f);

    Vector3f _position{0.f, 0.f, 0.f};
    bool _halfStructure{false};
    Scene &_scene;
    GlycansMap _glycans;
    ProteinMap _proteins;
    MeshMap _meshes;
    std::vector<std::pair<Vector3f, float>> _occupiedDirections;
    Vector4fs _clippingPlanes;
};
} // namespace bioexplorer
#endif // BIOEXPLORER_ASSEMBLY_H
