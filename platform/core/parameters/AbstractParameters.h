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
#include <platform/core/common/Logs.h>
#include <platform/core/common/Types.h>

#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>

namespace po = boost::program_options;

namespace core
{
/**
   Base class defining command line parameters
 */
class AbstractParameters : public BaseObject
{
public:
    /**
       Constructor
       @param name Display name for the set of parameters
     */
    AbstractParameters(const std::string& name)
        : _name(name)
        , _parameters(name)
    {
    }

    virtual ~AbstractParameters() = default;
    /**
       Parses parameters managed by the class
       @param vm the variables map of all arguments passed by the user
     */
    virtual void parse(const po::variables_map&){};

    /**
       Displays values of registered parameters
     */
    virtual void print();

    po::options_description& parameters() { return _parameters; }

protected:
    std::string _name;

    po::options_description _parameters;

    static std::string asString(const bool flag) { return flag ? "on" : "off"; }
};
} // namespace core

namespace boost
{
namespace program_options
{
/**
 * Wrapper for supporting fixed size multitoken values
 */
template <typename T, typename charT = char>
class fixed_tokens_typed_value : public typed_value<T, charT>
{
    const unsigned _min;
    const unsigned _max;

    typedef typed_value<T, charT> base;

public:
    fixed_tokens_typed_value(T* t, unsigned min, unsigned max)
        : base(t)
        , _min(min)
        , _max(max)
    {
        base::multitoken();
    }
    unsigned min_tokens() const { return _min; }
    unsigned max_tokens() const { return _max; }
};

template <typename T>
inline fixed_tokens_typed_value<T>* fixed_tokens_value(unsigned min, unsigned max)
{
    return new fixed_tokens_typed_value<T>(nullptr, min, max);
}

template <typename T>
inline fixed_tokens_typed_value<T>* fixed_tokens_value(T* t, unsigned min, unsigned max)
{
    return new fixed_tokens_typed_value<T>(t, min, max);
}
} // namespace program_options
} // namespace boost
