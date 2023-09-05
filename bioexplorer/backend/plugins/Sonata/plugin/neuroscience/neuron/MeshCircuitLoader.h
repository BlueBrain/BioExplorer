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

#include "AbstractCircuitLoader.h"

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
class MeshCircuitLoader : public AbstractCircuitLoader
{
public:
    MeshCircuitLoader(core::Scene &scene, const core::ApplicationParameters &applicationParameters,
                      core::PropertyMap &&loaderParams);

    std::string getName() const final;

    static core::PropertyMap getCLIProperties();

    core::ModelDescriptorPtr importFromFile(const std::string &filename, const core::LoaderProgress &callback,
                                            const core::PropertyMap &properties) const final;
};
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
