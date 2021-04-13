/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <plugin/common/Types.h>

namespace bioexplorer
{
/**
 * @brief The Assembly class is a container for biological entities (proteins,
 * membranes, sugars, etc.)
 */
class Assembly
{
public:
    /**
     * @brief Assembly Default constructor
     * @param scene Scene to which assembly should be added
     * @param assemblyDescriptor Description of the assembly
     */
    Assembly(Scene &scene, const AssemblyDescriptor &assemblyDescriptor);

    /**
     * @brief Destroy the Assembly object
     *
     */
    ~Assembly();

    /**
     * @brief setColorScheme Set a color scheme to a protein of the assembly
     * @param descriptor Color scheme descriptor
     */
    void setColorScheme(const ColorSchemeDescriptor &descriptor);

    /**
     * @brief setAminoAcidSequenceAsString Apply a color scheme to visualize a
     * given amino acid sequence defined by a string
     * @param descriptor Amino acid sequence as a string
     */
    void setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDescriptor &descriptor);

    /**
     * @brief setAminoAcidSequenceAsRange Apply a color scheme to visualize a
     * given amino acid sequence defined by a range of indices
     * @param descriptor Amino acid sequence as a range of indices
     */
    void setAminoAcidSequenceAsRange(
        const AminoAcidSequenceAsRangesDescriptor &descriptor);

    /**
     * @param descriptor Name of the assembly and name of the protein
     * @return Amino acid sequence and indices for a given protein of the
     * assembly
     */
    std::string getAminoAcidInformation(
        const AminoAcidInformationDescriptor &descriptor) const;

    /**
     * @brief Set an amino acid at a given position in the protein sequences
     *
     * @param aminoAcid Structure containing the information related the amino
     * acid to be modified
     */
    void setAminoAcid(const SetAminoAcid &aminoAcid);

    /**
     * @return Clipping planes applied to the assembly
     */
    Vector4fs &getClippingPlanes() { return _clippingPlanes; }

    /**
     * @brief setClippingPlanes Set clipping planes on the assembly
     * @param clippingPlanes Clipping planes as a vector of 4 floats
     */
    void setClippingPlanes(const Vector4fs &clippingPlanes)
    {
        _clippingPlanes = clippingPlanes;
    }

    /**
     * @return The description of the assembly
     */
    const AssemblyDescriptor &getDescriptor() { return _descriptor; }

    /**
     * @return A map of the proteins in the assembly
     */
    const ProteinMap &getProteins() const { return _proteins; }

    /**
     * @brief Set the transformation for a specific instance of a protein
     * @param descriptor Information about the instance
     */
    void setProteinInstanceTransformation(
        const ProteinInstanceTransformationDescriptor &descriptor);

    /**
     * @return The transformation for a specific instance of a protein
     */
    const Transformation getProteinInstanceTransformation(
        const ProteinInstanceTransformationDescriptor &descriptor) const;

    /**
     * @brief addMembrane Add a membrane to the assembly
     * @param descriptor Membrane descriptor
     */
    void addMembrane(const MembraneDescriptor &descriptor);

    /**
     * @brief addRNASequence Add an RNA sequence to the assembly
     * @param descriptor Descriptor of the RNA sequence
     */
    void addRNASequence(const RNASequenceDescriptor &descriptor);

    /**
     * @brief addProtein Add a protein to the assembly
     * @param descriptor Descriptor of the protein
     */
    void addProtein(const ProteinDescriptor &descriptor);

    /**
     * @brief addMeshBasedMembrane Add a mesh based membrane to the assembly
     * @param descriptor Descriptor of the mesh based membrane
     */
    void addMeshBasedMembrane(const MeshBasedMembraneDescriptor &descriptor);

    /**
     * @brief addGlycans Add glycans to glycosilation sites of a given protein
     * in the assembly
     * @param descriptor Descriptor of the glycans
     */
    void addGlycans(const SugarsDescriptor &descriptor);

    /**
     * @brief addSugars Add sugars to sugar binding sites of a given protein of
     * the assembly
     * @param descriptor Descriptor of the sugars
     */
    void addSugars(const SugarsDescriptor &descriptor);

private:
    void _processInstances(ModelDescriptorPtr md, const std::string &name,
                           const AssemblyShape shape,
                           const floats &assemblyParams,
                           const size_t occurrences, const Vector3f &position,
                           const Quaterniond &orientation,
                           const size_ts &allowedOccurrences,
                           const size_t randomSeed,
                           const PositionRandomizationType &randomizationType);

    AssemblyDescriptor _descriptor;
    Scene &_scene;
    ProteinMap _proteins;
    MeshBasedMembraneMap _meshBasedMembranes;
    MembranePtr _membrane{nullptr};
    RNASequencePtr _rnaSequence{nullptr};
    Vector4fs _clippingPlanes;
};
} // namespace bioexplorer
