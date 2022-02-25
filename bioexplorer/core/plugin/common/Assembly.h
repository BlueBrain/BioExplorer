/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue Brain Project / EPFL
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
#include <plugin/common/shapes/Shape.h>

namespace bioexplorer
{
namespace common
{
using namespace common;
using namespace details;
using namespace molecularsystems;
#ifdef USE_VASCULATURE
using namespace vasculature;
#endif
#ifdef USE_MORPHOLOGIES
using namespace morphology;
#endif

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
    void setProteinColorScheme(const ProteinColorSchemeDetails &details);

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
    Vector4ds &getClippingPlanes() { return _clippingPlanes; }

    /**
     * @brief setClippingPlanes Set clipping planes on the assembly
     * @param clippingPlanes Clipping planes as a vector of 4 doubles
     */
    void setClippingPlanes(const Vector4ds &clippingPlanes)
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
     * @return A pointer to a protein in the assembly
     */
    ProteinPtr getProtein(const std::string &name);

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
     * @brief addMembrane Add a membrane to the assembly
     * @param details Membrane details
     */
    void addMembrane(const MembraneDetails &details);

    /**
     * @brief Get the Membrane object
     *
     * @return const MembranePtr
     */
    const MembranePtr getMembrane() const { return _membrane; }

    /**
     * @brief addRNASequence Add an RNA sequence to the assembly
     * @param details Details of the RNA sequence
     */
    void addRNASequence(const RNASequenceDetails &details);

    const RNASequencePtr getRNASequence() { return _rnaSequence; }

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
    void addGlycan(const SugarDetails &details);

    /**
     * @brief addSugars Add sugars to sugar binding sites of a given protein of
     * the assembly
     * @param details Details of the sugars
     */
    void addSugar(const SugarDetails &details);

    /**
     * @brief addEnzymeReaction Add enzyme reaction to the scene
     *
     * @param details Details about the enzyme reaction
     * @param enzymeAssembly Pointer to the assembly containing the enzyme
     * @param enzyme Pointer to the enzyme
     * @param substrates List of pointers to the substrates
     * @param products List of pointers to the products
     */
    void addEnzymeReaction(const EnzymeReactionDetails &details,
                           AssemblyPtr enzymeAssembly, ProteinPtr enzyme,
                           Proteins &substrates, Proteins &products);

    /**
     * @brief setEnzymeReactionProgress Set enzyme reaction progress
     *
     * @param details Details about the enzyme reaction
     */
    void setEnzymeReactionProgress(
        const EnzymeReactionProgressDetails &details);

    /**
     * @brief Check if a location is inside the assembly
     *
     * @param point Location to check
     * @return true if the location is inside
     * @return false if the location is outside
     */
    bool isInside(const Vector3d &location) const;

    /**
     * @brief Returns information about the first protein hit by a ray defined
     * by an origin and a direction
     *
     * @param origin Origin of the ray
     * @param direction Direction of the ray
     * @return ProteinInspectionDetails Details about the protein
     */
    ProteinInspectionDetails inspect(const Vector3d &origin,
                                     const Vector3d &direction,
                                     double &t) const;

    /**
     * @brief Get the assembly transformation
     *
     * @return Transformation Assembly transformation
     */
    Transformation getTransformation() const;

    /**
     * @brief Get the assembly shape
     *
     * @return Shape Assembly shape
     */
    ShapePtr getShape() const { return _shape; }

#ifdef USE_VASCULATURE
    void addVasculature(const VasculatureDetails &details);
    std::string getVasculatureInfo() const;
    void setVasculatureColorScheme(
        const VasculatureColorSchemeDetails &payload);
    void setVasculatureReport(const VasculatureReportDetails &payload);
    void setVasculatureRadiusReport(
        const VasculatureRadiusReportDetails &details);
#endif

#ifdef USE_MORPHOLOGIES
    void addAstrocytes(const AstrocytesDetails &details);
#endif

private:
    void _processInstances(ModelDescriptorPtr md, const std::string &name,
                           const size_t occurrences, const Vector3d &position,
                           const Quaterniond &rotation,
                           const uint64_ts &allowedOccurrences,
                           const AnimationDetails &animationDetails,
                           const double offset,
                           const AssemblyConstraints &constraints);

    AssemblyDetails _details;
    Scene &_scene;
    ProteinMap _proteins;
    MembranePtr _membrane{nullptr};
    RNASequencePtr _rnaSequence{nullptr};
    EnzymeReactionMap _enzymeReactions;
    Vector3d _position;
    Quaterniond _rotation;
    Vector4ds _clippingPlanes;
    ModelDescriptors _modelDescriptors;
    ShapePtr _shape{nullptr};
#ifdef USE_VASCULATURE
    VasculaturePtr _vasculature{nullptr};
#endif
#ifdef USE_MORPHOLOGIES
    AstrocytesPtr _astrocytes{nullptr};
#endif
};
} // namespace common
} // namespace bioexplorer
