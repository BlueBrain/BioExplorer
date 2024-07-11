/*
    Copyright 2019 - 0211 Blue Brain Project / EPFL

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

#include <string>
#include <vector>

namespace core
{
namespace string_utils
{
std::string shortenString(const std::string& string, const size_t maxLength = 32);

std::string replaceFirstOccurrence(std::string input, const std::string& toReplace, const std::string& replaceWith);

std::string camelCaseToSeparated(const std::string& camelCase, const char separator);

std::string separatedToCamelCase(const std::string& separated, const char separator);

std::string join(const std::vector<std::string>& strings, const std::string& joinWith);

std::string toLowercase(const std::string input);

void trim(std::string& s);

std::vector<std::string> split(const std::string& s, char delim);
} // namespace string_utils
} // namespace core
