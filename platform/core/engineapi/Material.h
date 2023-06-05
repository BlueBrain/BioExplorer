/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
#include <platform/core/common/PropertyObject.h>
#include <platform/core/common/material/Texture2D.h>
#include <platform/core/common/MathTypes.h>
#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(Material)

namespace core
{
typedef std::map<TextureType, Texture2DPtr> TextureDescriptors;

class Material : public PropertyObject
{
public:
    /** @name API for engine-specific code */
    //@{
    /**
     * Called after material change
     */
    virtual void commit() = 0;
    //@}

    PLATFORM_API Material(const PropertyMap& properties = {});

    PLATFORM_API const std::string& getName() const { return _name; }
    PLATFORM_API void setName(const std::string& value) { _updateValue(_name, value); }
    PLATFORM_API void setDiffuseColor(const Vector3d& value) { _updateValue(_diffuseColor, value); }
    PLATFORM_API const Vector3d& getDiffuseColor() const { return _diffuseColor; }
    PLATFORM_API void setSpecularColor(const Vector3d& value) { _updateValue(_specularColor, value); }
    PLATFORM_API const Vector3d& getSpecularColor() const { return _specularColor; }
    PLATFORM_API void setSpecularExponent(const double value) { _updateValue(_specularExponent, value); }
    PLATFORM_API double getSpecularExponent() const { return _specularExponent; }
    PLATFORM_API void setReflectionIndex(const double value) { _updateValue(_reflectionIndex, value); }
    PLATFORM_API double getReflectionIndex() const { return _reflectionIndex; }
    PLATFORM_API void setOpacity(const double value) { _updateValue(_opacity, value); }
    PLATFORM_API double getOpacity() const { return _opacity; }
    PLATFORM_API void setRefractionIndex(const double value) { _updateValue(_refractionIndex, value); }
    PLATFORM_API double getRefractionIndex() const { return _refractionIndex; }
    PLATFORM_API void setEmission(double value) { _updateValue(_emission, value); }
    PLATFORM_API double getEmission() const { return _emission; }
    PLATFORM_API void setGlossiness(const double value) { _updateValue(_glossiness, value); }
    PLATFORM_API double getGlossiness() const { return _glossiness; }
    PLATFORM_API const TextureDescriptors& getTextureDescriptors() const { return _textureDescriptors; }
    PLATFORM_API void setTexture(const std::string& fileName, const TextureType type);
    PLATFORM_API void removeTexture(const TextureType type);
    PLATFORM_API void setShadingMode(const MaterialShadingMode value) { _updateValue(_shadingMode, value); }
    PLATFORM_API MaterialShadingMode getShadingMode() const { return _shadingMode; }
    PLATFORM_API void setUserParameter(const double value) { _updateValue(_userParameter, value); }
    PLATFORM_API double getUserParameter() const { return _userParameter; }
    PLATFORM_API void setCastUserData(const bool value) { _updateValue(_castUserData, value); }
    PLATFORM_API double getCastUserData() const { return _castUserData; }
    PLATFORM_API void setClippingMode(const MaterialClippingMode value) { _updateValue(_clippingMode, value); }
    PLATFORM_API MaterialClippingMode getClippingMode() const { return _clippingMode; }
    PLATFORM_API void setChameleonMode(const MaterialChameleonMode value) { _updateValue(_chameleonMode, value); }
    PLATFORM_API MaterialChameleonMode getChameleonMode() const { return _chameleonMode; }

    PLATFORM_API Texture2DPtr getTexture(const TextureType type) const;
    bool hasTexture(const TextureType type) const { return _textureDescriptors.count(type) > 0; }
    void clearTextures();

protected:
    bool _loadTexture(const std::string& fileName, const TextureType type);

    std::string _name{"undefined"};
    Vector3d _diffuseColor{1., 1., 1.};
    Vector3d _specularColor{1., 1., 1.};
    double _specularExponent{10.};
    double _reflectionIndex{0.};
    double _opacity{1.};
    double _refractionIndex{1.};
    double _emission{0.};
    double _glossiness{1.};
    TexturesMap _textures;
    TextureDescriptors _textureDescriptors;
    double _userParameter{1.};
    MaterialShadingMode _shadingMode{MaterialShadingMode::undefined_shading_mode};
    bool _castUserData{false};
    MaterialClippingMode _clippingMode{MaterialClippingMode::no_clipping};
    MaterialChameleonMode _chameleonMode{MaterialChameleonMode::undefined_chameleon_mode};

    SERIALIZATION_FRIEND(Material)
};
} // namespace core
