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

#include <platform/core/common/Types.h>

#include <functional>
#include <string>

namespace core
{
/**
 * Interface for registering actions, namely notifications which have no return
 * values with an optional parameter, and requests which return a value after
 * processing.
 *
 * The encoding of the parameter and return value is restricted to JSON.
 *
 * The parameters object must be deserializable by a free function:
 * @code
 * bool from_json(Params& object, const std::string& json)
 * @endcode
 *
 * The return type must be serializable by a free function:
 * @code
 * std::string to_json(const RetVal&)
 * @endcode
 */
class ActionInterface
{
public:
    virtual ~ActionInterface() = default;

    /**
     * Register an action with no parameter and no return value.
     *
     * @param desc description of the action/RPC
     * @param action the action to perform on an incoming notification
     */
    virtual void registerNotification(const RpcDescription& desc, const std::function<void()>& action) = 0;

    /**
     * Register an action with a property map as the parameter and no return
     * value.
     *
     * @param desc description of the action/RPC
     * @param input the acceptable property map as the parameter for the RPC
     * @param action the action to perform on an incoming notification
     */
    virtual void registerNotification(const RpcParameterDescription& desc, const PropertyMap& input,
                                      const std::function<void(PropertyMap)>& action) = 0;

    /**
     * Register an action with a property map as the parameter and a property
     * map as the return value.
     *
     * @param desc description of the action/RPC
     * @param input the acceptable property map as the parameter for the RPC
     * @param output the property map layout that is returned on a successful
     *               request
     * @param action the action to perform on an incoming request
     */
    virtual void registerRequest(const RpcParameterDescription& desc, const PropertyMap& input,
                                 const PropertyMap& output, const std::function<PropertyMap(PropertyMap)>& action) = 0;

    /**
     * Register an action with no parameter and a property map as the return
     * value.
     *
     * @param desc description of the action/RPC
     * @param output the property map layout that is returned on a successful
     *               request
     * @param action the action to perform on an incoming request
     */
    virtual void registerRequest(const RpcDescription& desc, const PropertyMap& output,
                                 const std::function<PropertyMap()>& action) = 0;

    /** Register an action with no parameter and no return value. */
    void registerNotification(const std::string& name, const std::function<void()>& action)
    {
        _registerNotification(name, [action] { action(); });
    }

    /** Register an action with a parameter and no return value. */
    template <typename Params>
    void registerNotification(const std::string& name, const std::function<void(Params)>& action)
    {
        _registerNotification(name,
                              [action](const std::string& param)
                              {
                                  Params params;
                                  if (!from_json(params, param))
                                      throw std::runtime_error("from_json failed");
                                  action(params);
                              });
    }

    /** Register an action with a parameter and a return value. */
    template <typename Params, typename RetVal>
    void registerRequest(const std::string& name, const std::function<RetVal(Params)>& action)
    {
        _registerRequest(name,
                         [action](const std::string& param)
                         {
                             Params params;
                             if (!from_json(params, param))
                                 throw std::runtime_error("from_json failed");
                             return to_json(action(params));
                         });
    }

    /** Register an action with no parameter and a return value. */
    template <typename RetVal>
    void registerRequest(const std::string& name, const std::function<RetVal()>& action)
    {
        _registerRequest(name, [action] { return to_json(action()); });
    }

protected:
    using RetParamFunc = std::function<std::string(std::string)>;
    using RetFunc = std::function<std::string()>;
    using ParamFunc = std::function<void(std::string)>;
    using VoidFunc = std::function<void()>;

private:
    virtual void _registerRequest(const std::string&, const RetParamFunc&) {}
    virtual void _registerRequest(const std::string&, const RetFunc&) {}
    virtual void _registerNotification(const std::string&, const ParamFunc&) {}
    virtual void _registerNotification(const std::string&, const VoidFunc&) {}
};
} // namespace core
