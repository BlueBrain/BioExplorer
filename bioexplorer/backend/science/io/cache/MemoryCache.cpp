/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "MemoryCache.h"

#include <science/common/GeneralSettings.h>
#include <science/common/Logs.h>
#include <science/io/db/DBConnector.h>
#include <science/io/filesystem/MorphologyLoader.h>

namespace bioexplorer
{
namespace io
{
using namespace common;
using namespace morphology;
using namespace details;
using namespace db;
using namespace filesystem;

MemoryCache* MemoryCache::_instance = nullptr;
std::mutex MemoryCache::_mutex;

const morphology::SectionMap& MemoryCache::getNeuronSections(const uint64_t neuronId, const NeuronsDetails& details)
{
    if (_enabled)
    {
        const auto it = _sections.find(neuronId);
        if (it != _sections.end())
            return (*it).second;
    }

    if (GeneralSettings::getInstance()->getLoadMorphologiesFromFileSystem())
    {
        NeuronSectionTypes sectionTypes;
        if (details.loadAxon)
            sectionTypes.push_back(NeuronSectionType::axon);
        if (details.loadApicalDendrites)
            sectionTypes.push_back(NeuronSectionType::apical_dendrite);
        if (details.loadBasalDendrites)
            sectionTypes.push_back(NeuronSectionType::basal_dendrite);

        const auto morphologyLoader = _getMorphologyLoader(details.populationName);
        const auto path = DBConnector::getInstance().getNeuronMorphologyRelativePath(details.populationName, neuronId);
        _sections[neuronId] = morphologyLoader->getNeuronSections(path, sectionTypes);
    }
    else
    {
        _sections[neuronId] =
            DBConnector::getInstance().getNeuronSections(details.populationName, neuronId, details.sqlSectionFilter);
    }
    return _sections[neuronId];
}

void MemoryCache::setEnabled(const bool enabled)
{
    _enabled = enabled;
    PLUGIN_INFO(1, "Memory cache is " << (_enabled ? "ON" : "OFF"));
    if (!enabled)
        _sections.clear();
}

MorphologyLoaderPtr MemoryCache::_getMorphologyLoader(const std::string& populationName)
{
    const auto itm = _morphologyLoaders.find(populationName);
    if (itm != _morphologyLoaders.end())
        return (*itm).second;

    const auto& connector = DBConnector::getInstance();
    const auto configurationValues = connector.getNeuronConfiguration(populationName);

    const auto itc = configurationValues.find(NEURON_CONFIG_MORPHOLOGY_FOLDER);
    if (itc == configurationValues.end())
        PLUGIN_THROW(NEURON_CONFIG_MORPHOLOGY_FOLDER + " is not defined in the configuration table");

    const auto itf = configurationValues.find(NEURON_CONFIG_MORPHOLOGY_FILE_EXTENSION);
    if (itf == configurationValues.end())
        PLUGIN_THROW(NEURON_CONFIG_MORPHOLOGY_FILE_EXTENSION + " is not defined in the configuration table");

    _morphologyLoaders[populationName] =
        std::shared_ptr<MorphologyLoader>(new MorphologyLoader((*itc).second, (*itf).second));
    return _morphologyLoaders[populationName];
}
} // namespace io
} // namespace bioexplorer
