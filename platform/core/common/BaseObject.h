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

#include <atomic>
#include <cmath>
#include <functional>
#include <type_traits>

namespace core
{
class BaseObject
{
public:
    BaseObject() = default;
    virtual ~BaseObject() = default;

    /**
     * Custom copy constructor to not copy changedCallback and solve
     * non-copyable atomic modified state.
     */
    BaseObject(const BaseObject&)
        : _modified(true)
    {
    }

    /** Custom assignment operator that does not copy the changedCallback. */
    BaseObject& operator=(const BaseObject& rhs)
    {
        if (this == &rhs)
            return *this;

        _modified = true;
        return *this;
    }

    /**
     * @return true if any parameter has been modified since the last
     *         resetModified().
     */
    bool isModified() const { return _modified; }
    /**
     * Reset the modified state, typically done after changes have been applied.
     */
    void resetModified() { _modified = false; }
    void markModified(const bool triggerCallback = true)
    {
        _modified = true;
        if (_modifiedCallback && triggerCallback)
            _modifiedCallback(*this);
    }

    using ModifiedCallback = std::function<void(const BaseObject&)>;

    /**
     * Set a function that is called after this object has been modified.
     */
    void onModified(const ModifiedCallback& callback) { _modifiedCallback = callback; }

    void clearModifiedCallback() { _modifiedCallback = ModifiedCallback(); }

protected:
    /**
     * Helper function for derived classes to update a parameter and mark it
     * modified if it has changed.
     */
    template <typename T>
    void _updateValue(T& member, const T& newValue, const bool triggerCallback = true)
    {
        if (!_isEqual(member, newValue))
        {
            member = newValue;
            markModified(triggerCallback);
        }
    }

    template <class T>
    bool _isEqual(const T& a, const T& b, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
    {
        return std::fabs(a - b) < 0.000001;
    }

    template <class T>
    bool _isEqual(const T& a, const T& b, typename std::enable_if<!std::is_floating_point<T>::value>::type* = 0)
    {
        return a == b;
    }

private:
    std::atomic_bool _modified{true};
    ModifiedCallback _modifiedCallback;
};
} // namespace core
