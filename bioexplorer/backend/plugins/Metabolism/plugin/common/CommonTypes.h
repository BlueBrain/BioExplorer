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

enum SpatialLocation
{
    neuron = 0,
    astrocyte = 1,
    extra_cellular_space = 2,
    capillarity = 3,
    synaptic_area = 4,
    neuron_mitochondria = 5,
    astrocyte_mitochondria = 6
};
