/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "XYZBLoader.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/utils/FileSystem.h>
#include <platform/core/common/utils/StringUtils.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include <fstream>
#include <sstream>

namespace core
{
namespace
{
constexpr auto ALMOST_ZERO = 1e-7f;
constexpr auto LOADER_NAME = "xyzb";

float _computeHalfArea(const Boxf& bbox)
{
    const auto size = bbox.getSize();
    return size[0] * size[1] + size[0] * size[2] + size[1] * size[2];
}
} // namespace

XYZBLoader::XYZBLoader(Scene& scene)
    : Loader(scene)
{
}

bool XYZBLoader::isSupported(const std::string& storage, const std::string& extension) const
{
    const std::set<std::string> types = {"xyz"};
    return types.find(extension) != types.end();
}

ModelDescriptorPtr XYZBLoader::importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                              const PropertyMap& properties) const
{
    CORE_INFO("Loading xyz " << blob.name);

    std::stringstream stream(std::string(blob.data.begin(), blob.data.end()));
    size_t numlines = 0;
    {
        numlines = std::count(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>(), '\n');
    }
    stream.seekg(0);

    auto model = _scene.createModel();

    const auto name = fs::path({blob.name}).stem();
    const auto materialId = 0;
    model->createMaterial(materialId, name);
    auto& spheres = model->getSpheres()[materialId];

    const size_t startOffset = spheres.size();
    spheres.reserve(spheres.size() + numlines);

    Boxf bbox;
    size_t i = 0;
    std::string line;
    std::stringstream msg;
    msg << "Loading " << string_utils::shortenString(blob.name) << " ...";
    while (std::getline(stream, line))
    {
        std::vector<float> lineData;
        std::stringstream lineStream(line);

        float value;
        while (lineStream >> value)
            lineData.push_back(value);

        switch (lineData.size())
        {
        case 3:
        {
            const Vector3f position(lineData[0], lineData[1], lineData[2]);
            bbox.merge(position);
            // The point radius used here is irrelevant as it's going to be
            // changed later.
            model->addSphere(materialId, {position, 1});
            break;
        }
        default:
            throw std::runtime_error("Invalid content in line " + std::to_string(i + 1) + ": " + line);
        }
        callback.updateProgress(msg.str(), i++ / static_cast<float>(numlines));
    }

    // Find an appropriate mean radius to avoid overlaps of the spheres, see
    // https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_radius

    const auto volume = glm::compMul(bbox.getSize());
    const auto density4PI = 4 * M_PI * numlines / (volume > ALMOST_ZERO ? volume : _computeHalfArea(bbox));

    const double meanRadius = volume > ALMOST_ZERO ? std::pow((3. / density4PI), 1. / 3.) : std::sqrt(1 / density4PI);

    // resize the spheres to the new mean radius
    for (i = 0; i < numlines; ++i)
        spheres[i + startOffset].radius = meanRadius;

    Transformation transformation;
    transformation.setRotationCenter(model->getBounds().getCenter());
    auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), blob.name);
    modelDescriptor->setTransformation(transformation);

    Property radiusProperty("radius", meanRadius, 0., meanRadius * 2., {"Point size"});
    radiusProperty.onModified(
        [modelDesc = std::weak_ptr<ModelDescriptor>(modelDescriptor)](const auto& property)
        {
            if (auto modelDesc_ = modelDesc.lock())
            {
                const auto newRadius = property.template get<double>();
                for (auto& sphere : modelDesc_->getModel().getSpheres()[materialId])
                    sphere.radius = newRadius;
            }
        });
    PropertyMap modelProperties;
    modelProperties.setProperty(radiusProperty);
    modelDescriptor->setProperties(modelProperties);
    return modelDescriptor;
}

ModelDescriptorPtr XYZBLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                 const PropertyMap& properties) const
{
    std::ifstream file(storage);
    if (!file.good())
        CORE_THROW("Could not open file " + storage);
    return importFromBlob({"xyz", storage, {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()}},
                          callback, properties);
}

std::string XYZBLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> XYZBLoader::getSupportedStorage() const
{
    return {"xyz"};
}
} // namespace core
