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

#pragma once

#include <platform/core/common/loader/Loader.h>
#include <platform/core/parameters/GeometryParameters.h>

struct aiScene;

namespace core
{
/** Loads meshes from files using the assimp library
 * http://assimp.sourceforge.net
 */
class MeshLoader : public Loader
{
public:
    MeshLoader(Scene& scene);
    MeshLoader(Scene& scene, const GeometryParameters& geom);

    std::vector<std::string> getSupportedStorage() const final;
    std::string getName() const final;
    PropertyMap getProperties() const final;

    bool isSupported(const std::string& storage, const std::string& extension) const final;

    ModelDescriptorPtr importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                         const PropertyMap& properties) const final;

    ModelDescriptorPtr importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                      const PropertyMap& properties) const final;

    ModelMetadata importMesh(const std::string& fileName, const LoaderProgress& callback, Model& model,
                             const Matrix4f& transformation, const size_t defaultMaterialId,
                             const GeometryQuality geometryQuality) const;

private:
    PropertyMap _defaults;

    void _createMaterials(Model& model, const aiScene* aiScene, const std::string& folder) const;

    ModelMetadata _postLoad(const aiScene* aiScene, Model& model, const Matrix4f& transformation,
                            const size_t defaultMaterial, const std::string& folder,
                            const LoaderProgress& callback) const;
    size_t _getQuality(const GeometryQuality geometryQuality) const;
};
} // namespace core
