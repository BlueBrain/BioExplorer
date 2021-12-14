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
namespace biology
{
using namespace details;

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
     * @param details Details of the assembly
     */
    Assembly(Scene &scene, const AssemblyDetails &details);

    /**
     * @brief Destroy the Assembly object
     *
     */
    ~Assembly();

    /**
     * @brief setColorScheme Set a color scheme to a protein of the assembly
     * @param details Color scheme details
     */
    void setColorScheme(const ColorSchemeDetails &details);

    /**
     * @brief setAminoAcidSequenceAsString Apply a color scheme to visualize a
     * given amino acid sequence defined by a string
     * @param details Amino acid sequence as a string
     */
    void setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDetails &details);

    /**
     * @brief setAminoAcidSequenceAsRange Apply a color scheme to visualize a
     * given amino acid sequence defined by a range of indices
     * @param details Amino acid sequence as a range of indices
     */
    void setAminoAcidSequenceAsRange(
        const AminoAcidSequenceAsRangesDetails &details);

    /**
     * @param details Name of the assembly and name of the protein
     * @return Amino acid sequence and indices for a given protein of the
     * assembly
     */
    const std::string getAminoAcidInformation(
        const AminoAcidInformationDetails &details) const;

    /**
     * @brief Set an amino acid at a given position in the protein sequences
     *
     * @param details Structure containing the information related the amino
     * acid to be modified
     */
    void setAminoAcid(const AminoAcidDetails &details);

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
    const AssemblyDetails &getDescriptor() { return _details; }

    /**
     * @return A map of the proteins in the assembly
     */
    const ProteinMap &getProteins() const { return _proteins; }

    /**
     * @brief Set the transformation for a specific instance of a protein
     * @param details Details about the instance
     */
    void setProteinInstanceTransformation(
        const ProteinInstanceTransformationDetails &details);

    /**
     * @param details Details about the instance
     * @return The transformation for a specific instance of a protein
     */
    const Transformation getProteinInstanceTransformation(
        const ProteinInstanceTransformationDetails &details) const;

    /**
     * @brief addParametricMembrane Add a parametric membrane to the assembly
     * @param details Parametric membrane details
     */
    void addParametricMembrane(const ParametricMembraneDetails &details);

    /**
     * @brief addMeshBasedMembrane Add a mesh based membrane to the assembly
     * @param details Details of the mesh based membrane
     */
    void addMeshBasedMembrane(const MeshBasedMembraneDetails &details);

    /**
     * @brief addRNASequence Add an RNA sequence to the assembly
     * @param details Details of the RNA sequence
     */
    void addRNASequence(const RNASequenceDetails &details);

    /**
     * @brief addProtein Add a protein to the assembly
     * @param details Details of the protein
     */
    void addProtein(const ProteinDetails &details,
                    const AssemblyConstraints &constraints);

    /**
     * @brief addGlycans Add glycans to glycosilation sites of a given protein
     * in the assembly
     * @param details Details of the glycans
     */
    void addGlycans(const SugarsDetails &details);

    /**
     * @brief addSugars Add sugars to sugar binding sites of a given protein of
     * the assembly
     * @param details Details of the sugars
     */
    void addSugars(const SugarsDetails &details);

    bool isInside(const Vector3f &point) const;

private:
    void _processInstances(ModelDescriptorPtr md, const std::string &name,
                           const AssemblyShape shape,
                           const floats &assemblyParams,
                           const size_t occurrences, const Vector3f &position,
                           const Quaterniond &orientation,
                           const size_ts &allowedOccurrences,
                           const size_t randomSeed,
                           const PositionRandomizationType &randomizationType,
                           const AssemblyConstraints &constraints);

    AssemblyDetails _details;
    Scene &_scene;
    ProteinMap _proteins;
    MembranePtr _membrane{nullptr};
    RNASequencePtr _rnaSequence{nullptr};
    Vector3f _position;
    Quaterniond _rotation;
    Vector4fs _clippingPlanes;
    ModelDescriptors _modelDescriptors;
};
} // namespace biology
} // namespace bioexplorer
