/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include <core/brayns/common/Transformation.h>
#include <core/brayns/common/loader/Loader.h>
#include <core/brayns/common/Types.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace astrocyte
{
using namespace brayns;

class AstrocyteLoader : public Loader
{
public:
    AstrocyteLoader(Scene &scene, const ApplicationParameters &applicationParameters, PropertyMap &&loaderParams);

    std::string getName() const final;

    std::vector<std::string> getSupportedExtensions() const final;

    bool isSupported(const std::string &filename, const std::string &extension) const final;

    static PropertyMap getCLIProperties();

    /** @copydoc Loader::importFromBlob */
    ModelDescriptorPtr importFromBlob(Blob &&blob, const LoaderProgress &callback,
                                      const PropertyMap &properties) const final;

    /** @copydoc Loader::importFromFile */
    ModelDescriptorPtr importFromFile(const std::string &filename, const LoaderProgress &callback,
                                      const PropertyMap &properties) const final;

private:
    void _importMorphologiesFromURIs(const PropertyMap &properties, const std::vector<std::string> &uris,
                                     const LoaderProgress &callback, Model &model) const;
    const ApplicationParameters &_applicationParameters;
    PropertyMap _defaults;
    PropertyMap _fixedDefaults;
};
} // namespace astrocyte
} // namespace neuroscience
} // namespace sonataexplorer
