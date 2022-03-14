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
using namespace details;
using namespace common;
#ifdef USE_MORPHOLOGIES
using namespace morphology;
#endif

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

#ifdef USE_VASCULATURE
    /**
     * @brief Get the population ID from a given name
     *
     * @param populationName Name of the population
     * @return Id of the population
     */
    uint64_t getVasculaturePopulationId(
        const std::string& populationName) const;

    /**
     * @brief Get the Nodes for a given population
     *
     * @param populationId Id of the population
     * @return GeometryNodes
     */
    GeometryNodes getVasculatureNodes(const std::string& populationName,
                                      const std::string& filter = "",
                                      const double scale = 1.0) const;

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
     * @param simulationReportId Simulation report identifier
     * @return SimulationReport Information about the simulation Report
     */

    SimulationReport getVasculatureSimulationReport(
        const std::string& populationName,
        const int32_t simulationReportId) const;

    /**
     * @brief Get time series from simulation report
     *
     * @param simulationReportId Simulation report identifier
     * @param frame Frame number
     * @return floats Values of the simulation frame
     */
    floats getVasculatureSimulationTimeSeries(const int32_t simulationReportId,
                                              const int32_t frame) const;
#endif

#ifdef USE_MORPHOLOGIES
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
     * @brief Get the end-feet areas of a given astrocyte
     *
     * @param astrocyteId Identifier of the astrocyte
     * @return SectionMap A map of end-feet
     */
    EndFootMap getAstrocyteEndFeetAreas(const uint64_t astrocyteId) const;

    /**
     * @brief Get the neurons locations
     *
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return NeuronSomaMap A map of neurons (position, type, etc)
     */
    NeuronSomaMap getNeurons(const std::string& sqlCondition = "") const;

    /**
     * @brief Get the sections of a given neuron
     *
     * @param neuronId Identifier of the neuron
     * @param sqlCondition String containing an WHERE condition for the SQL
     * statement
     * @return SectionMap A map of sections
     */
    SectionMap getNeuronSections(const int64_t neuronId,
                                 const std::string& sqlCondition = "") const;

#endif

    static std::mutex _mutex;
    static DBConnector* _instance;

private:
    DBConnector();
    ~DBConnector();

    ConnectionPtr _connection{nullptr};
};

} // namespace db
} // namespace io
} // namespace bioexplorer
