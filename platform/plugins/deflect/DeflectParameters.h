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

#include <platform/core/common/BaseObject.h>
#include <platform/core/common/PropertyMap.h>

#include <deflect/types.h>

namespace core
{
constexpr auto PARAM_CHROMA_SUBSAMPLING = "chromaSubsampling";
constexpr auto PARAM_COMPRESSION = "compression";
constexpr auto PARAM_ENABLED = "enabled";
constexpr auto PARAM_HOSTNAME = "hostname";
constexpr auto PARAM_ID = "id";
constexpr auto PARAM_PORT = "port";
constexpr auto PARAM_QUALITY = "quality";
constexpr auto PARAM_RESIZING = "resizing";
constexpr auto PARAM_TOP_DOWN = "topDown";
constexpr auto PARAM_USE_PIXEL_OP = "usePixelop";

class DeflectParameters : public BaseObject
{
public:
    DeflectParameters();

    bool getEnabled() const { return _props.getProperty<bool>(PARAM_ENABLED); }
    void setEnabled(const bool enabled) { _updateProperty(PARAM_ENABLED, enabled); }
    bool getCompression() const { return _props.getProperty<bool>(PARAM_COMPRESSION); }
    void setCompression(const bool enabled) { _updateProperty(PARAM_COMPRESSION, enabled); }

    unsigned getQuality() const { return (unsigned)_props.getProperty<int32_t>(PARAM_QUALITY); }
    void setQuality(const unsigned quality) { _updateProperty(PARAM_QUALITY, (int32_t)quality); }
    std::string getId() const { return _props.getProperty<std::string>(PARAM_ID); }
    void setId(const std::string& id) { _updateProperty(PARAM_ID, id); }
    std::string getHostname() const { return _props.getProperty<std::string>(PARAM_HOSTNAME); }
    void setHost(const std::string& host) { _updateProperty(PARAM_HOSTNAME, host); }

    unsigned getPort() const { return (unsigned)_props.getProperty<int32_t>(PARAM_PORT); }
    void setPort(const unsigned port) { _updateProperty(PARAM_PORT, (int32_t)port); }

    bool isResizingEnabled() const { return _props.getProperty<bool>(PARAM_RESIZING); }

    bool isTopDown() const { return _props.getProperty<bool>(PARAM_TOP_DOWN); }
    void setIsTopDown(const bool topDown) { _updateProperty(PARAM_TOP_DOWN, topDown); }

    bool usePixelOp() const { return _props.getProperty<bool>(PARAM_USE_PIXEL_OP); }
    deflect::ChromaSubsampling getChromaSubsampling() const
    {
        return (deflect::ChromaSubsampling)_props.getProperty<int32_t>(PARAM_CHROMA_SUBSAMPLING);
    }
    void setChromaSubsampling(const deflect::ChromaSubsampling subsampling)
    {
        _updateProperty(PARAM_CHROMA_SUBSAMPLING, (int32_t)subsampling);
    }

    const PropertyMap& getPropertyMap() const { return _props; }
    PropertyMap& getPropertyMap() { return _props; }

private:
    PropertyMap _props;

    template <typename T>
    void _updateProperty(const char* property, const T& newValue)
    {
        if (!_isEqual(_props.getProperty<T>(property), newValue))
        {
            _props.updateProperty(property, newValue);
            markModified();
        }
    }
};
} // namespace core
