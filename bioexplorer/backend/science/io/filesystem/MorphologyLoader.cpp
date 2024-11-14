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

#include "MorphologyLoader.h"

#include <science/common/Logs.h>

#ifdef USE_MORPHIO
#include <morphio/morphology.h>
#include <morphio/section.h>
#endif // USE_MORPHIO

namespace bioexplorer
{
namespace io
{
namespace filesystem
{
using namespace core;
using namespace morphology;
using namespace details;

#ifdef USE_MORPHIO
using namespace morphio;
#endif

MorphologyLoader::MorphologyLoader(const std::string& morphologyFolder, const std::string& morphologyFileExtension)
    : _morphologyFolder(morphologyFolder)
    , _morphologyFileExtension(morphologyFileExtension)
{
}

#ifdef USE_MORPHIO
void loadSection(const morphio::Section& section, morphology::SectionMap& sections,
                 const NeuronSectionTypes& sectionTypes)
{
    if (std::find(sectionTypes.begin(), sectionTypes.end(), static_cast<NeuronSectionType>(section.type())) ==
        sectionTypes.end())
        return;

    morphology::Section s;
    s.type = section.type();
    s.parentId = section.isRoot() ? SOMA_AS_PARENT : section.parent().id();

    const auto& p = section.points();
    const auto& d = section.diameters();
    s.points.resize(p.size());
    for (uint64_t i = 0; i < p.size(); ++i)
        s.points[i] = Vector4f(p[i][0], p[i][1], p[i][2], d[i]);

    s.length = 0.0;
    for (uint64_t i = 0; i < s.points.size() - 1; ++i)
        s.length += length(Vector3f(s.points[i + 1]) - Vector3f(s.points[i]));

    sections[section.id()] = s;

    for (const auto& child : section.children())
        loadSection(child, sections, sectionTypes);
}
#endif

SectionMap MorphologyLoader::getNeuronSections(const std::string& filename,
                                               const NeuronSectionTypes& sectionTypes) const
{
#ifdef USE_MORPHIO
    morphology::SectionMap sections;
    const auto path = _morphologyFolder + "/" + filename + "." + _morphologyFileExtension;
    Morphology morph = Morphology(path, Option::NRN_ORDER);
    for (const auto& section : morph.rootSections())
        loadSection(section, sections, sectionTypes);
    return sections;
#else
    PLUGIN_THROW("BioExplorer was not compiled with MorphIO");
#endif
}
} // namespace filesystem
} // namespace io
} // namespace bioexplorer
