/* Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of the circuit explorer for Brayns
 * <https://github.com/favreau/Brayns-UC-CircuitExplorer>
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

#include <common/log.h>

#include "BioExplorerLoader.h"

#include <api/BioExplorerParams.h>
#include <common/Assembly.h>
#include <common/Protein.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>

#include <fstream>

namespace bioexplorer
{
const std::string LOADER_NAME = "Bio Explorer loader";
const std::string SUPPORTED_EXTENTION_BIOEXPLORER = "bioexplorer";

const Property PROP_LOAD_VIRUSES = {"loadViruses",
                                    bool(true),
                                    {"Load viruses"}};
const Property PROP_LOAD_CELLS = {"loadCells", bool(true), {"Load cells"}};
const Property PROP_LOAD_SPDS = {"loadSPDs",
                                 bool(true),
                                 {"Load D-surfactants"}};
const Property PROP_LOAD_GLUCOSE = {"loadGlucose",
                                    bool(true),
                                    {"Load glucose"}};
const Property PROP_LOAD_DEFENSINS = {"loadDefensins",
                                      bool(true),
                                      {"Load defensins"}};

BioExplorerLoader::BioExplorerLoader(Scene& scene, PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
    PLUGIN_INFO << "Registering " << LOADER_NAME << std::endl;
}

std::string BioExplorerLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> BioExplorerLoader::getSupportedExtensions() const
{
    return {SUPPORTED_EXTENTION_BIOEXPLORER};
}

bool BioExplorerLoader::isSupported(const std::string& /*filename*/,
                                    const std::string& extension) const
{
    const std::set<std::string> types = {SUPPORTED_EXTENTION_BIOEXPLORER};
    return types.find(extension) != types.end();
}

ModelDescriptorPtr BioExplorerLoader::importFromBlob(
    Blob&& /*blob*/, const LoaderProgress& /*callback*/,
    const PropertyMap& /*properties*/) const
{
    throw std::runtime_error(
        "Loading molecular systems from blob is not supported");
}

ModelDescriptorPtr BioExplorerLoader::importFromFile(
    const std::string& filename, const LoaderProgress& callback,
    const PropertyMap& properties) const
{
    auto model = _scene.createModel();

    PropertyMap props = _defaults;
    props.merge(properties);

    return nullptr;
}

void BioExplorerLoader::exportToFile(const std::string& filename,
                                     const AssemblyMap& assemblies) const
{
    PLUGIN_INFO << "Saving scene to bioexplorer file: " << filename
                << std::endl;
    std::ofstream file(filename, std::ios::out);
    if (!file.good())
    {
        const std::string msg = "Could not open bioexplorer file " + filename;
        PLUGIN_THROW(std::runtime_error(msg));
    }

    for (const auto& assembly : assemblies)
    {
        file << "{assembly:";
        const auto& ad = assembly.second->getDescriptor();
        const auto& s = to_json(ad);
        file << s;

        file << ", proteins:[";
        const auto& proteins = assembly.second->getProteins();
        bool first = true;
        for (const auto& protein : proteins)
        {
            if (!first)
                file << ",";
            file << "protein:";
            const auto& pd = protein.second->getDescriptor();
            const auto& s = to_json(pd);
            file << s;
            first = false;
        }
        file << "]"; // Proteins

        file << "}"; // Assembly
    }
    file.close();
}

PropertyMap BioExplorerLoader::getProperties() const
{
    return _defaults;
}

PropertyMap BioExplorerLoader::getCLIProperties()
{
    PropertyMap pm("BioExplorerLoader");
    pm.setProperty(PROP_LOAD_VIRUSES);
    pm.setProperty(PROP_LOAD_CELLS);
    pm.setProperty(PROP_LOAD_SPDS);
    pm.setProperty(PROP_LOAD_GLUCOSE);
    pm.setProperty(PROP_LOAD_DEFENSINS);
    return pm;
}
} // namespace bioexplorer
