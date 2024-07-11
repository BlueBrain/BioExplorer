/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include <brain/brain.h>
#include <brion/brion.h>

#include <string>

namespace sonataexplorer
{
// Command line arguments
// - Database
const std::string ARG_DB_HOST = "--db-host";
const std::string ARG_DB_PORT = "--db-port";
const std::string ARG_DB_NAME = "--db-name";
const std::string ARG_DB_USER = "--db-user";
const std::string ARG_DB_PASSWORD = "--db-password";
const std::string ARG_DB_NB_CONNECTIONS = "--db-nb-connections";
const std::string ARG_DB_BATCH_SIZE = "--db-batch-size";
const size_t DEFAULT_DB_NB_CONNECTIONS = 8;
} // namespace sonataexplorer
