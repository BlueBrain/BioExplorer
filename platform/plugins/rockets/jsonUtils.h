/*
    Copyright 2015 - 2018 Blue Brain Project / EPFL

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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "staticjson/staticjson.hpp"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "rapidjson/document.h"
#include "rapidjson/writer.h"

namespace core
{
/** @return JSON schema from JSON-serializable type */
template <class T>
std::string buildJsonSchema(const std::string& title)
{
    T obj;
    return buildJsonSchema(obj, title);
}

/** @return JSON schema from JSON-serializable object */
template <class T>
std::string buildJsonSchema(T& obj, const std::string& title)
{
    using namespace rapidjson;
    auto schema = staticjson::export_json_schema(&obj);
    schema.AddMember(StringRef("title"), StringRef(title.c_str()), schema.GetAllocator());

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    schema.Accept(writer);
    return buffer.GetString();
}

/** @return JSON schema for JSON RPC parameter */
template <class T>
rapidjson::Document getRPCParameterSchema(const std::string& paramName, const std::string& paramDescription, T& obj)
{
    using namespace rapidjson;
    auto schema = staticjson::export_json_schema(&obj);
    auto& allocator = schema.GetAllocator();

    schema.AddMember(StringRef("name"), Value(paramName.c_str(), allocator), allocator);
    schema.AddMember(StringRef("description"), Value(paramDescription.c_str(), allocator), allocator);
    return schema;
};

rapidjson::Document _buildJsonRpcSchema(const RpcDescription& desc)
{
    using namespace rapidjson;
    Document schema(kObjectType);
    auto& allocator = schema.GetAllocator();
    schema.AddMember(StringRef("title"), StringRef(desc.methodName.c_str()), allocator);
    schema.AddMember(StringRef("description"), StringRef(desc.methodDescription.c_str()), allocator);
    schema.AddMember(StringRef("type"), StringRef("method"), allocator);
    schema.AddMember(StringRef("async"), desc.type == Execution::async, allocator);
    return schema;
}

/**
 * @return JSON schema for RPC with one parameter and a return value, according
 * to
 * http://www.simple-is-better.org/json-rpc/jsonrpc20-schema-service-descriptor.html
 */
template <class P, class R>
std::string buildJsonRpcSchemaRequest(const RpcParameterDescription& desc, P& obj)
{
    using namespace rapidjson;
    auto schema = _buildJsonRpcSchema(desc);
    auto& allocator = schema.GetAllocator();

    R retVal;
    auto retSchema = staticjson::export_json_schema(&retVal);
    schema.AddMember(StringRef("returns"), retSchema, allocator);

    Value params(kArrayType);
    auto paramSchema = getRPCParameterSchema<P>(desc.paramName, desc.paramDescription, obj);
    params.PushBack(paramSchema, allocator);
    schema.AddMember(StringRef("params"), params, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    schema.Accept(writer);
    return buffer.GetString();
}

template <class P, class R>
std::string buildJsonRpcSchemaRequest(const RpcParameterDescription& desc)
{
    P obj;
    return buildJsonRpcSchemaRequest<P, R>(desc, obj);
}

/**
 * @return JSON schema for RPC with no parameter, but a return value, according
 * to
 * http://www.simple-is-better.org/json-rpc/jsonrpc20-schema-service-descriptor.html
 */
template <class R>
std::string buildJsonRpcSchemaRequestReturnOnly(const RpcDescription& desc, R& retVal)
{
    using namespace rapidjson;
    auto schema = _buildJsonRpcSchema(desc);
    auto& allocator = schema.GetAllocator();

    auto retSchema = staticjson::export_json_schema(&retVal);
    schema.AddMember(StringRef("returns"), retSchema, allocator);

    Value params(kArrayType);
    schema.AddMember(StringRef("params"), params, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    schema.Accept(writer);
    return buffer.GetString();
}

template <class R>
std::string buildJsonRpcSchemaRequestReturnOnly(const RpcDescription& desc)
{
    R retVal;
    return buildJsonRpcSchemaRequestReturnOnly<R>(desc, retVal);
}

/**
 * @return JSON schema for RPC with one parameter and no return value, according
 * to
 * http://www.simple-is-better.org/json-rpc/jsonrpc20-schema-service-descriptor.html
 */
template <class P>
std::string buildJsonRpcSchemaNotify(const RpcParameterDescription& desc, P& obj)
{
    using namespace rapidjson;
    auto schema = _buildJsonRpcSchema(desc);
    auto& allocator = schema.GetAllocator();

    Value params(kArrayType);
    auto paramSchema = getRPCParameterSchema<P>(desc.paramName, desc.paramDescription, obj);
    params.PushBack(paramSchema, allocator);
    schema.AddMember(StringRef("params"), params, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    schema.Accept(writer);
    return buffer.GetString();
}

template <class P>
std::string buildJsonRpcSchemaNotify(const RpcParameterDescription& desc)
{
    P obj;
    return buildJsonRpcSchemaNotify<P>(desc, obj);
}

/** @return JSON schema for RPC with no parameter and no return value. */
std::string buildJsonRpcSchemaNotify(const RpcDescription& desc)
{
    using namespace rapidjson;
    auto schema = _buildJsonRpcSchema(desc);
    auto& allocator = schema.GetAllocator();

    schema.AddMember(StringRef("returns"), Value(kNullType), allocator);

    Value params(kArrayType);
    schema.AddMember(StringRef("params"), params, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    schema.Accept(writer);
    return buffer.GetString();
}
} // namespace core
