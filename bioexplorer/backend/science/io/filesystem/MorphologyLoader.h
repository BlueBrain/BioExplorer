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

#pragma once

#include <science/common/Types.h>

namespace bioexplorer
{
namespace io
{
namespace filesystem
{
/**
 * Load molecular systems from an optimized binary representation of the 3D
 * scene
 */
class MorphologyLoader
{
public:
    /**
     * @brief Construct a new object
     *
     * @param morphologyFolder Morphology folder
     */
    MorphologyLoader(const std::string& morphologyFolder, const std::string& morphologyFileExtension);

    /**
     * @brief Import sections from morphology file
     *
     * @param filename Full path of the morphology file
     * @param sectionTypes Section types to import
     * @return SectionMap Sections from the morphology file
     */
    morphology::SectionMap getNeuronSections(const std::string& filename,
                                             const details::NeuronSectionTypes& sectionTypes) const;

private:
    std::string _morphologyFolder;
    std::string _morphologyFileExtension;
};
} // namespace filesystem
} // namespace io
} // namespace bioexplorer
