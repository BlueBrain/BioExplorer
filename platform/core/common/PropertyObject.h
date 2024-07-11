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
#include <platform/core/common/Types.h>

#include <map>

namespace core
{
/**
 * Maps generic properties to user-defined types/keys/names and tracks the
 * current type/key/name for querying, setting and updating its properties.
 */
class PropertyObject : public BaseObject
{
public:
    /** Set the current type to use for 'type-less' queries and updates. */
    void setCurrentType(const std::string& type)
    {
        _updateValue(_currentType, type);

        // add default (empty) property map for new type
        if (_properties.count(type) == 0)
            _properties[type];
    }

    /** @return the current set type. */
    const std::string& getCurrentType() const { return _currentType; }
    /** Update the value of the given property for the current type. */
    template <typename T>
    inline void updateProperty(const std::string& name, const T& value, const bool triggerCallback = true)
    {
        auto& propMap = _properties.at(_currentType);
        const auto oldValue = propMap.getProperty<T>(name, value);
        if (!_isEqual(oldValue, value))
        {
            propMap.updateProperty(name, value);
            markModified(triggerCallback);
        }
    }

    /**
     * @return true if the property with the given name exists for the current
     *         type.
     */
    bool hasProperty(const std::string& name) const { return _properties.at(_currentType).hasProperty(name); }

    /**
     * @return the value of the property with the given name for the current
     *         type.
     */
    template <typename T>
    inline T getProperty(const std::string& name) const
    {
        return _properties.at(_currentType).getProperty<T>(name);
    }

    /**
     * @return the value of the property with the given name for the current
     *         type. If it does not exist return the given value.
     */
    template <typename T>
    inline T getPropertyOrValue(const std::string& name, T val) const
    {
        return hasProperty(name) ? getProperty<T>(name) : val;
    }

    /** Assign a new set of properties to the current type. */
    void setProperties(const PropertyMap& properties) { setProperties(_currentType, properties); }

    /** Assign a new set of properties to the given type. */
    void setProperties(const std::string& type, const PropertyMap& properties)
    {
        _properties[type] = properties;
        markModified();
    }

    /**
     * Update or add all the properties from the given map to the current type.
     */
    void updateProperties(const PropertyMap& properties)
    {
        _properties.at(_currentType).merge(properties);
        markModified();
    }

    /** @return the entire property map for the current type. */
    const auto& getPropertyMap() const { return _properties.at(_currentType); }
    /** @return the entire property map for the given type. */
    const auto& getPropertyMap(const std::string& type) const { return _properties.at(type); }

    /** @return the list of all registered types. */
    strings getTypes() const
    {
        strings types;
        for (const auto& i : _properties)
            types.push_back(i.first);
        return types;
    }

    /** Clear all current properties and clone new properties from object  */
    void clonePropertiesFrom(const PropertyObject& obj)
    {
        _currentType = obj._currentType;
        _properties.clear();
        for (const auto& kv : obj._properties)
        {
            const auto& key = kv.first;
            const auto& properties = kv.second.getProperties();

            PropertyMap propertyMapClone;
            for (const auto& property : properties)
                propertyMapClone.setProperty(*property);

            _properties[key] = propertyMapClone;
        }
    }

protected:
    std::string _currentType;
    std::map<std::string, PropertyMap> _properties;
};
} // namespace core
