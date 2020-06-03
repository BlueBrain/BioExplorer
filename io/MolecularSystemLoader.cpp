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

#include "MolecularSystemLoader.h"

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>

#include <fstream>

namespace bioexplorer

{
const std::string LOADER_NAME = "Molecular system loader";
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

MolecularSystemLoader::MolecularSystemLoader(Scene& scene,
                                             PropertyMap&& loaderParams)
    : Loader(scene)
    , _defaults(loaderParams)
{
    PLUGIN_INFO << "Registering " << LOADER_NAME << std::endl;
}

std::string MolecularSystemLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> MolecularSystemLoader::getSupportedExtensions() const
{
    return {SUPPORTED_EXTENTION_BIOEXPLORER};
}

bool MolecularSystemLoader::isSupported(const std::string& /*filename*/,
                                        const std::string& extension) const
{
    const std::set<std::string> types = {SUPPORTED_EXTENTION_BIOEXPLORER};
    return types.find(extension) != types.end();
}

ModelDescriptorPtr MolecularSystemLoader::importFromBlob(
    Blob&& /*blob*/, const LoaderProgress& /*callback*/,
    const PropertyMap& /*properties*/) const
{
    throw std::runtime_error(
        "Loading molecular systems from blob is not supported");
}

ModelDescriptorPtr MolecularSystemLoader::importFromFile(
    const std::string& filename, const LoaderProgress& callback,
    const PropertyMap& properties) const
{
    auto model = _scene.createModel();

    PropertyMap props = _defaults;
    props.merge(properties);

    return nullptr;
}

PropertyMap MolecularSystemLoader::getProperties() const
{
    return _defaults;
}

PropertyMap MolecularSystemLoader::getCLIProperties()
{
    PropertyMap pm("MolecularSystemLoader");
    pm.setProperty(PROP_LOAD_VIRUSES);
    pm.setProperty(PROP_LOAD_CELLS);
    pm.setProperty(PROP_LOAD_SPDS);
    pm.setProperty(PROP_LOAD_GLUCOSE);
    pm.setProperty(PROP_LOAD_DEFENSINS);
    return pm;
}
} // namespace bioexplorer
