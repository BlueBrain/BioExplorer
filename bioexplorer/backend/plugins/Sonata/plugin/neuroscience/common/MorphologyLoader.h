/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <plugin/api/SonataExplorerParams.h>

#include <plugin/neuroscience/common/Types.h>

#include <platform/core/common/Types.h>
#include <platform/core/common/loader/Loader.h>
#include <platform/core/parameters/GeometryParameters.h>

#include <vector>

namespace sonataexplorer
{
namespace neuroscience
{
namespace common
{
struct MorphologyTreeStructure
{
    std::vector<int> sectionParent;
    std::vector<std::vector<size_t>> sectionChildren;
    std::vector<size_t> sectionTraverseOrder;
};

/** Loads morphologies from SWC and H5, and Circuit Config files */
class MorphologyLoader : public core::Loader
{
public:
    MorphologyLoader(core::Scene& scene, core::PropertyMap&& loaderParams,
                     const core::Transformation& transformation = core::Transformation());

    /** @copydoc Loader::getName */
    std::string getName() const final;

    /** @copydoc Loader::getSupportedStorage */
    strings getSupportedStorage() const final;

    /** @copydoc Loader::isSupported */
    bool isSupported(const std::string& storage, const std::string& extension) const final;

    /** @copydoc Loader::getCLIProperties */
    static core::PropertyMap getCLIProperties();

    /** @copydoc Loader::getProperties */
    core::PropertyMap getProperties() const final;

    /** @copydoc Loader::importFromBlob */
    core::ModelDescriptorPtr importFromBlob(core::Blob&& blob, const core::LoaderProgress& callback,
                                            const core::PropertyMap& properties) const final;

    /** @copydoc Loader::importFromFile */
    core::ModelDescriptorPtr importFromStorage(const std::string& storage, const core::LoaderProgress& callback,
                                               const core::PropertyMap& properties) const final;

    /**
     * @brief importMorphology imports a single morphology from a specified URI
     * @param uri URI of the morphology
     * @param index Index of the morphology
     * @param defaultMaterialId Material to use
     * @param compartmentReport Compartment report to map to the morphology
     * @return Model container
     */
    ParallelModelContainer importMorphology(const Gid& gid, const core::PropertyMap& properties,
                                            const std::string& source, const uint64_t index,
                                            const SynapsesInfo& synapsesInfo,
                                            const core::Matrix4f& transformation = core::Matrix4f(),
                                            CompartmentReportPtr compartmentReport = nullptr,
                                            const float mitochondriaDensity = 0.f) const;

    /**
     * @brief setBaseMaterialId Set the base material ID for the morphology
     * @param materialId Id of the base material ID for the morphology
     */
    void setBaseMaterialId(const size_t materialId) { _baseMaterialId = materialId; }

    /**
     * @brief createMissingMaterials Checks that all materials exist for
     * existing geometry in the model. Missing materials are created with the
     * default parameters
     */
    static void createMissingMaterials(core::Model& model, const bool castUserData = false);

    static const brain::neuron::SectionTypes getSectionTypesFromProperties(const core::PropertyMap& properties);

private:
    /**
     * @brief _getCorrectedRadius Modifies the diameter of the geometry
     * according to --radius-multiplier and --radius-correction geometry
     * parameters
     * @param diameter Diameter to be corrected and converted in to radius
     * @return Corrected value of a radius according to geometry parameters
     */
    float _getCorrectedRadius(const core::PropertyMap& properties, const float diameter) const;

    void _importMorphology(const Gid& gid, const core::PropertyMap& properties, const std::string& source,
                           const uint64_t index, const core::Matrix4f& transformation,
                           common::ParallelModelContainer& model, CompartmentReportPtr compartmentReport,
                           const SynapsesInfo& synapsesInfo, const float mitochondriaDensity = 0.f) const;

    /**
     * @brief _importMorphologyAsPoint places sphere at the specified morphology
     * position
     * @param index Index of the current morphology
     * @param material Material that is forced in case geometry parameters do
     * not apply
     * @param compartmentReport Compartment report to map to the morphology
     * @param scene Scene to which the morphology should be loaded into
     */
    void _importMorphologyAsPoint(const core::PropertyMap& properties, const uint64_t index,
                                  common::CompartmentReportPtr compartmentReport,
                                  common::ParallelModelContainer& model) const;

    /**
     * @brief _importMorphologyFromURI imports a morphology from the specified
     * URI
     * @param uri URI of the morphology
     * @param index Index of the current morphology
     * @param materialFunc A function mapping brain::neuron::SectionType to a
     * material id
     * @param compartmentReport Compartment report to map to the morphology
     * @param model Model container to whichh the morphology should be loaded
     * into
     */
    void _importMorphologyFromURI(const Gid& gid, const core::PropertyMap& properties, const std::string& uri,
                                  const uint64_t index, const core::Matrix4f& transformation,
                                  CompartmentReportPtr compartmentReport, ParallelModelContainer& model,
                                  const SynapsesInfo& synapsesInfo, const float mitochondriaDensity) const;

    size_t _addSDFGeometry(SDFMorphologyData& sdfMorphologyData, const core::SDFGeometry& geometry,
                           const std::set<size_t>& neighbours, const size_t materialId, const int section) const;

    /**
     * Creates an SDF soma by adding and connecting the soma children using cone
     * pills
     */
    void _connectSDFSomaChildren(const core::PropertyMap& properties, const core::Vector3f& somaPosition,
                                 const float somaRadius, const size_t materialId, const uint64_t& userDataOffset,
                                 const brain::neuron::Sections& somaChildren,
                                 SDFMorphologyData& sdfMorphologyData) const;

    /**
     * Goes through all bifurcations and connects to all connected SDF
     * geometries it is overlapping. Every section that has a bifurcation will
     * traverse its children and blend the geometries inside the bifurcation.
     */
    void _connectSDFBifurcations(SDFMorphologyData& sdfMorphologyData, const MorphologyTreeStructure& mts) const;

    /**
     * Calculates all neighbours and adds the geometries to the model container.
     */
    void _finalizeSDFGeometries(ParallelModelContainer& modelContainer, SDFMorphologyData& sdfMorphologyData) const;

    /**
     * Calculates the structure of the morphology tree by finding overlapping
     * beginnings and endings of the sections.
     */
    MorphologyTreeStructure _calculateMorphologyTreeStructure(const core::PropertyMap& properties,
                                                              const brain::neuron::Sections& sections) const;

    /**
     * Adds a Soma geometry to the model
     */
    void _addSomaGeometry(const uint64_t index, const core::PropertyMap& properties, const brain::neuron::Soma& soma,
                          uint64_t offset, ParallelModelContainer& model, SDFMorphologyData& sdfMorphologyData,
                          const bool useSimulationModel, const bool generateInternals, const float mitochondriaDensity,
                          uint32_t& sdfGroupId) const;

    /**
     * Adds the sphere between the steps in the sections
     */
    void _addStepSphereGeometry(const bool useSDFGeometry, const bool isDone, const core::Vector3f& position,
                                const float radius, const size_t materialId, const uint64_t& userDataOffset,
                                ParallelModelContainer& model, SDFMorphologyData& sdfMorphologyData,
                                const uint32_t sdfGroupId, const float displacementRatio = 1.f) const;

    /**
     * Adds the cone between the steps in the sections
     */
    void _addStepConeGeometry(const bool useSDFGeometry, const core::Vector3f& source, const float sourceRadius,
                              const core::Vector3f& target, const float targetRadius, const size_t materialId,
                              const uint64_t& userDataOffset, ParallelModelContainer& model,
                              SDFMorphologyData& sdfMorphologyData, const uint32_t sdfGroupId,
                              const float displacementRatio = 1.f) const;

    /**
     * @brief _getMaterialIdFromColorScheme returns the material id
     * corresponding to the morphology color scheme and the section type
     * @param sectionType Section type of the morphology
     * @return Material Id
     */
    size_t _getMaterialIdFromColorScheme(const core::PropertyMap& properties,
                                         const brain::neuron::SectionType& sectionType) const;

    /**
     * @brief Computes the distance of a segment to the soma
     * @param section Section containing the segment
     * @param sampleId segment index in the section
     * @return Distance to the soma
     */
    float _distanceToSoma(const brain::neuron::Section& section, const size_t sampleId) const;

    void _addSynapse(const core::PropertyMap& properties, const brain::Synapse& synapse, const SynapseType synapseType,
                     const brain::neuron::Sections& sections, const core::Vector3f& somaPosition,
                     const float somaRadius, const core::Matrix4f& transformation, const size_t materialId,
                     ParallelModelContainer& model, SDFMorphologyData& sdfMorphologyData, uint32_t& sdfGroupId) const;

    void _addSomaInternals(const core::PropertyMap& properties, const uint64_t index, ParallelModelContainer& model,
                           const size_t materialId, const float somaRadius, const float mitochondriaDensity,
                           SDFMorphologyData& sdfMorphologyData, uint32_t& sdfGroupId) const;

    void _addAxonInternals(const core::PropertyMap& properties, const float sectionLength, const float sectionVolume,
                           const core::Vector4fs& samples, const float mitochondriaDensity, const size_t materialId,
                           SDFMorphologyData& sdfMorphologyData, uint32_t& sdfGroupId,
                           ParallelModelContainer& model) const;

    void _addAxonMyelinSheath(const core::PropertyMap& properties, const float sectionLength,
                              const core::Vector4fs& samples, const float mitochondriaDensity, const size_t materialId,
                              SDFMorphologyData& sdfMorphologyData, uint32_t& sdfGroupId,
                              ParallelModelContainer& model) const;

    size_t _getNbMitochondrionSegments() const;

    size_t _baseMaterialId{core::NO_MATERIAL};
    core::PropertyMap _defaults;
    core::Transformation _transformation;
};
typedef std::shared_ptr<MorphologyLoader> MorphologyLoaderPtr;

} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
