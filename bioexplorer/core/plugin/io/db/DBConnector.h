/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <brayns/common/types.h>

#include <pqxx/pqxx>

namespace bioexplorer
{
namespace io
{
namespace db
{
using namespace brayns;
using namespace details;
using namespace common;
using namespace morphology;
using namespace connectomics;

using ConnectionPtr = std::shared_ptr<pqxx::connection>;

/**
 * @brief The DBConnector class allows the BioExplorer to communicate with a
 * PostgreSQL database. The DBConnector requires the pqxx library to be found at
 * compilation time.
 *
 */
class DBConnector
{
public:
    /**
     * @brief Get the Instance object
     *
     * @return GeneralSettings* Pointer to the object
     */
    static DBConnector& getInstance()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_instance)
            _instance = new DBConnector();
        return *_instance;
    }

    /**
     * @brief Connects to the database using the provided command line arguments
     *
     * @param arguments Command line arguments
     */
    void init(const CommandLineArguments& arguments);

    /**
     * @brief Get the number of connections to the database
     *
     * @return size_t Number of connections to the database
     */
    size_t getNbConnections() const { return _dbNbConnections; }

    /**
     * @brief Remove all bricks from the PostgreSQL database
     *
     */
    void clearBricks();

    /**
     * @brief Get the Scene configuration
     *
     * @return Configuration of the out-of-code read-only scene
     */
    const OOCSceneConfigurationDetails getSceneConfiguration();

    /**
     * @brief Get the Brick object
     *
     * @param brickId Identifier of the brick
     * @param version Version of the binary buffer with the contents of the
     * brick
     * @param nbModels The number of models contained in the brick
     * @return std::stringstream The binary buffer with the contents of the
     * brick
     */
    std::stringstream getBrick(const int32_t brickId, const uint32_t& version,
                               uint32_t& nbModels);

    /**
     * @brief Inserts a brick into the PostgreSQL database
     *
     * @param brickId Identifier of the brick
     * @param version Version of the binary buffer with the contents of the
     * brick
     * @param nbModels The number of models contained in the brick
     * @param buffer The binary buffer with the contents of the brick
     */
    void insertBrick(const int32_t brickId, const uint32_t version,
                     const uint32_t nbModels, const std::stringstream& buffer);

    /**
     * @brief Get the Nodes for a given population
     *
     * @param populationName Name of the population
     * @param filter SQL condition
     * @return GeometryNodes Vasculature nodes
     */
    GeometryNodes getVasculatureNodes(const std::string& populationName,
                                      const std::string& filter = "",
                                      const std::string& limits = "") const;

    /**
     * @brief Get the sections for a given population
     *
     * @param populationName Name of the population
     * @param filter SQL condition
     * @return Section ids
     */
    uint64_ts getVasculatureSections(const std::string& populationName,
                                     const std::string& filter = "");

    /**
     * @brief Get the number of sections for a given population
     *
     * @param populationName Name of the population
     * @param filter SQL condition
     * @return Number of sections, total number of sections
     */
    Vector2ui getVasculatureNbSections(const std::string& populationName,
                                       const std::string& filter = "");

    /**
     * @brief Get the Vasculature radius range
     *
     * @param populationName Name of the population
     * @param filter SQL condition
     * @return Vector2d Min and max radius for the node selection
     */
    Vector2d getVasculatureRadiusRange(const std::string& populationName,
                                       const std::string& filter) const;

    /**
     * @brief Get the Edges for a given population
     *
     * @param populationName Name of the population
     * @return EdgeNodes
     */
    GeometryEdges getVasculatureEdges(const std::string& populationName,
                                      const std::string& filter = "") const;

    /**
     * @brief Get the bifurcations for a given population
     *
     * @param populationName Name of the population
     * @return Bifurcations
     */
    Bifurcations getVasculatureBifurcations(
        const std::string& populationName) const;

    /**
     * @brief Get information about the simulation Report
     *
     * @param populationName Name of the population
     * @param simulationReportId Simulation report identifier
     * @return SimulationReport Information about the simulation Report
     */

    SimulationReport getSimulationReport(
        const std::string& populationName,
        const int32_t simulationReportId) const;

    /**
     * @brief Get time series from simulation report
     *
     * @param populationName Name of the population
     * @param simulationReportId Simulation report identifier
     * @param frame Frame number
     * @return floats Values of the simulation frame
     */
    floats getVasculatureSimulationTimeSeries(const std::string& populationName,
                                              const int32_t simulationReportId,
                                              const int32_t frame) const;

    /**
     * @brief Get the astrocytes locations
     *
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return SomaMap A map of somas (position, radius, etc)
     */
    AstrocyteSomaMap getAstrocytes(const std::string& sqlCondition = "") const;

    /**
     * @brief Get the sections of a given astrocyte
     *
     * @param astrocyteId Identifier of the astrocyte
     * @return SectionMap A map of sections
     */
    SectionMap getAstrocyteSections(const int64_t astrocyteId) const;

    /**
     * @brief Get the end-feet as nodes for a given astrocyte
     *
     * @param astrocyteId Identifier of the astrocyte
     * @return EndFootNodesMap A map of end-feet
     */
    EndFootMap getAstrocyteEndFeet(const std::string& vasculaturePopulationName,
                                   const uint64_t astrocyteId) const;

    /**
     * @brief Get the neurons locations
     *
     * @param populationName Name of the population
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return NeuronSomaMap A map of neurons (position, type, etc)
     */
    NeuronSomaMap getNeurons(const std::string& populationName,
                             const std::string& sqlCondition = "") const;

    /**
     * @brief Get the sections of a given neuron
     *
     * @param populationName Name of the population
     * @param neuronId Identifier of the neuron
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return SectionMap A map of sections
     */
    SectionMap getNeuronSections(const std::string& populationName,
                                 const uint64_t neuronId,
                                 const std::string& sqlCondition = "") const;

    /**
     * @brief Get the synapses attached to a given neuron
     *
     * @param populationName Name of the population
     * @param neuronId Identifier of the neuron
     * @param SynapseType Type of synapses (afferent or efferent)
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return SynapseMap A map of synapses
     */
    SynapseMap getNeuronSynapses(const std::string& populationName,
                                 const uint64_t neuronId,
                                 const SynapseType synapseType,
                                 const std::string& sqlCondition = "") const;

    /**
     * @brief Get the neuron report type
     *
     * @param populationName Name of the population
     * @param reportId Report identifier
     * @return ReportType Type of report
     */
    ReportType getNeuronReportType(const std::string& populationName,
                                   const uint64_t reportId) const;

    /**
     * @brief Get a selection of spikes from a neuron spike report
     *
     * @param populationName Name of the population
     * @param reportId Simulation report identifier
     * @param startTime Start time of the selection
     * @param endTime End time of the selection
     * @return uint64_ts Spiking neuron ids for the specified time selection
     */
    uint64_ts getNeuronSpikeReportValues(const std::string& populationName,
                                         const uint64_t reportId,
                                         const double startTime,
                                         const double endTime) const;

    /**
     * @brief Get the Neuron soma simulation values
     *
     * @param populationName Name of the population
     * @param reportId Simulation report identifier
     * @param frame Simulation frame
     * @return floats The Neuron soma simulation values
     */
    floats getNeuronSomaReportValues(const std::string& populationName,
                                     const uint64_t reportId,
                                     const uint64_t frame) const;

    /**
     * @brief Get the regions from the brain atlas
     *
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return A vector of regions Ids
     */
    uint64_ts getAtlasRegions(const std::string& sqlCondition = "") const;

    /**
     * @brief Get the cells from the brain atlas
     *
     * @param regionId Region identifier
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return CellMap A map of cells (position, orientation, type, etc)
     */
    CellMap getAtlasCells(const uint64_t regionId,
                          const std::string& sqlCondition = "") const;

    /**
     * @brief Get the mesh of a given region from the brain atlas
     *
     * @param regionId Region identifier
     * statement
     * @return TrianglesMesh A triangles mesh
     */
    TriangleMesh getAtlasMesh(const uint64_t regionId) const;

    /**
     * @brief Get the White Matter streamlines for a given population
     *
     * @param populationName Name of the population
     * @param filter SQL condition
     * @return WhiteMatterStreamlines White matter streamlines
     */
    WhiteMatterStreamlines getWhiteMatterStreamlines(
        const std::string& populationName,
        const std::string& filter = "") const;

    static std::mutex _mutex;
    static DBConnector* _instance;

private:
    DBConnector();
    ~DBConnector();

    size_t _dbNbConnections{DEFAULT_DB_NB_CONNECTIONS};

    std::string _connectionString;

    std::vector<ConnectionPtr> _connections;
    bool _initialized{false};
};

} // namespace db
} // namespace io
} // namespace bioexplorer
