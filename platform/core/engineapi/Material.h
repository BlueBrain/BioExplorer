/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#pragma once

#include <platform/core/common/Api.h>
#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/MathTypes.h>
#include <platform/core/common/PropertyObject.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/material/Texture2D.h>

SERIALIZATION_ACCESS(Material)

namespace core
{
typedef std::map<TextureType, Texture2DPtr> TextureDescriptors;

/**
 * @class Material
 * @extends PropertyObject
 * @brief The class that represents the material object.
 * This class is derived from PropertyObject and provides all relevant properties for a Material object.
 * It also provides functions to work with textures.
 */

class Material : public PropertyObject
{
public:
    /**
     * @brief Called after material change.
     */
    PLATFORM_API virtual void commit() = 0;

    /**
     * @brief Constructs a Material object.
     * @param properties The PropertyMap object to initialize the Material object with.
     */
    PLATFORM_API Material(const PropertyMap& properties = {});

    /**
     *  @brief Returns the name of the material.
     */
    PLATFORM_API const std::string& getName() const { return _name; }

    /**
     * @brief Sets the name of the material.
     * @param value The value to set the name to.
     */
    PLATFORM_API void setName(const std::string& value) { _updateValue(_name, value); }

    /**
     * @brief Sets the color of the diffuse component of the material.
     * @param value The Vector3d object representing the color.
     * */
    PLATFORM_API void setDiffuseColor(const Vector3d& value) { _updateValue(_diffuseColor, value); }
    /**
     * @brief Returns the color of the diffuse component of the material.
     */
    PLATFORM_API const Vector3d& getDiffuseColor() const { return _diffuseColor; }

    /**
     * @brief Sets the color of the specular component of the material.
     * @param value The Vector3d object representing the color.
     * */
    PLATFORM_API void setSpecularColor(const Vector3d& value) { _updateValue(_specularColor, value); }

    /**
     * @brief Returns the color of the specular component of the material.
     */
    PLATFORM_API const Vector3d& getSpecularColor() const { return _specularColor; }

    /**
     * @brief Sets the specular exponent of the material.
     */
    PLATFORM_API void setSpecularExponent(const double value) { _updateValue(_specularExponent, value); }
    /**
     * @brief Returns the specular exponent of the material.
     * @param value The specular exponent to set.
     */
    PLATFORM_API double getSpecularExponent() const { return _specularExponent; }

    /**
     * @brief Sets the reflection index of the material.
     * @param value The reflection index to set.
     */
    PLATFORM_API void setReflectionIndex(const double value) { _updateValue(_reflectionIndex, value); }
    /**
     * @brief Returns the reflection index of the material.
     */
    PLATFORM_API double getReflectionIndex() const { return _reflectionIndex; }

    /**
     * @brief Sets the opacity of the material.
     * @param value The opacity to set.
     */
    PLATFORM_API void setOpacity(const double value) { _updateValue(_opacity, value); }
    /**
     * @brief Returns the opacity of the material.
     */
    PLATFORM_API double getOpacity() const { return _opacity; }

    /**
     * @brief Sets the refraction index of the material.
     * @param value The refraction index to set.
     */
    PLATFORM_API void setRefractionIndex(const double value) { _updateValue(_refractionIndex, value); }
    /**
     * @brief Returns the refraction index of the material.
     */
    PLATFORM_API double getRefractionIndex() const { return _refractionIndex; }

    /**
     * @brief Sets the emission of the material.
     * @param value The emission to set.
     */
    PLATFORM_API void setEmission(double value) { _updateValue(_emission, value); }
    /**
     *  @brief Returns the emission of the material.
     */
    PLATFORM_API double getEmission() const { return _emission; }

    /**
     * @brief Sets the glossiness of the material.
     * @param value The glossiness to set.
     */
    PLATFORM_API void setGlossiness(const double value) { _updateValue(_glossiness, value); }
    /**
     * @brief Returns the glossiness of the material.
     */
    PLATFORM_API double getGlossiness() const { return _glossiness; }

    /**
     * @brief Sets the shading mode of the material
     * @param value The shading mode to set.
     */
    PLATFORM_API void setShadingMode(const MaterialShadingMode value) { _updateValue(_shadingMode, value); }
    /**
     * @brief Returns the shading mode of the material
     */
    PLATFORM_API MaterialShadingMode getShadingMode() const { return _shadingMode; }

    /**
     * @brief Sets the user parameter of the material
     * @param value The user parameter to set.
     */
    PLATFORM_API void setUserParameter(const double value) { _updateValue(_userParameter, value); }
    /**
     * @brief Returns the user parameter of the material
     */
    PLATFORM_API double getUserParameter() const { return _userParameter; }

    /**
     * @brief Sets the cast user data of the material
     * @param The value to set for the cast user data
     */
    PLATFORM_API void setCastUserData(const bool value) { _updateValue(_castUserData, value); }
    /**
     * @brief Returns the cast user data of the material
     */
    PLATFORM_API double getCastUserData() const { return _castUserData; }

    /**
     * @brief Sets the clipping mode of the material
     * @param value The clipping mode to set.
     */
    PLATFORM_API void setClippingMode(const MaterialClippingMode value) { _updateValue(_clippingMode, value); }
    /**
     * @brief Returns the clipping mode of the material
     */
    PLATFORM_API MaterialClippingMode getClippingMode() const { return _clippingMode; }

    /**
     * @brief Sets the chameleon mode of the material
     * @param value The chameleon mode to set.
     */
    PLATFORM_API void setChameleonMode(const MaterialChameleonMode value) { _updateValue(_chameleonMode, value); }
    /**
     * @brief Returns the chameleon mode of the material
     */
    PLATFORM_API MaterialChameleonMode getChameleonMode() const { return _chameleonMode; }

    /**
     * @brief Sets the cast user data of the material
     * @param The value to set for the cast user data
     */
    PLATFORM_API void setNodeId(const int32_t value) { _updateValue(_nodeId, value); }
    /**
     * @brief Returns the cast user data of the material
     */
    PLATFORM_API int32_t getNodeId() const { return _nodeId; }

    /**
     * @brief Returns the texture descriptors of the material.
     */
    PLATFORM_API const TextureDescriptors& getTextureDescriptors() const { return _textureDescriptors; }

    /**
     * @brief Sets the texture of the material for the specified texture type.
     */
    PLATFORM_API void setTexture(const std::string& fileName, const TextureType type);

    /**
     * @brief Removes the texture of the material for the specified texture type.
     */
    PLATFORM_API void removeTexture(const TextureType type);

    /**
     * @brief Returns the texture of the material for the specified texture type.
     * @param type The texture type to get.
     * @return The Texture2D object of the material for the specified texture type.
     */
    PLATFORM_API Texture2DPtr getTexture(const TextureType type) const;

    /**
     * @brief Checks if the material has a texture for the specified texture type.
     * @param type The texture type to check.
     * @return bool Whether or not the material has a texture for the specified texture type.
     */
    PLATFORM_API bool hasTexture(const TextureType type) const { return _textureDescriptors.count(type) > 0; }

    /**
     * @brief Clears all textures from the material object.
     */
    PLATFORM_API void clearTextures();

protected:
    /**
     * @brief Loads the texture for the specified texture type.
     * @param fileName The file name of the texture to load.
     * @param type The texture type to load it for.
     * @return bool whether or not the texture was loaded successfully.
     */
    bool _loadTexture(const std::string& fileName, const TextureType type);

    std::string _name{"undefined"};         // The name of the material.
    Vector3d _diffuseColor{1., 1., 1.};     // The color of the diffuse component of the material.
    Vector3d _specularColor{1., 1., 1.};    // The color of the specular component of the material.
    double _specularExponent{10.};          // The specular exponent of the material.
    double _reflectionIndex{0.};            // The reflection index of the material.
    double _opacity{1.};                    // The opacity of the material.
    double _refractionIndex{1.};            // The refraction index of the material.
    double _emission{0.};                   // The emission of the material.
    double _glossiness{1.};                 // The glossiness of the material.
    TexturesMap _textures;                  // The textures of the material.
    TextureDescriptors _textureDescriptors; // The texture descriptors of the material.
    double _userParameter{1.};              // The user parameter of the material.
    MaterialShadingMode _shadingMode{MaterialShadingMode::undefined_shading_mode}; // The shading mode of the material
    bool _castUserData{false};                                                     // The cast user data of the material
    MaterialClippingMode _clippingMode{MaterialClippingMode::no_clipping};         // The clipping mode of the material
    int32_t _nodeId;                                                               // ID attached to the material
    MaterialChameleonMode _chameleonMode{MaterialChameleonMode::undefined_chameleon_mode}; // The chameleon mode of the
                                                                                           // material

    SERIALIZATION_FRIEND(Material)
};
} // namespace core
