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

#include <stdint.h>

namespace bioexplorer
{
namespace common
{
/**
 * @brief The UniqueID class provides a way to get a unique identifier accross
 * the application
 *
 */
class UniqueId
{
protected:
    static uint32_t nextId;

public:
    /**
     * @brief Construct a new UniqueId object
     *
     */
    UniqueId();

    /**
     * @brief Get a unique identifier
     *
     * @return uint32_t
     */
    static uint32_t get();
};
} // namespace common
} // namespace bioexplorer
