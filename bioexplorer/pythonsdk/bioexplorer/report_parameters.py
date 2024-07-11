# Copyright 2020 - 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .math_utils import Vector2


class NeuronReportParams:
    """
    Parameters used to map report data to the neurons. These settings determine how neuron data
    is represented and handled within simulations.
    """

    def __init__(
        self,
        report_id=-1,
        value_range=Vector2(-80.0, 10.0),
        voltage_scaling_range=Vector2(1.0, 1.0),
        initial_simulation_frame=0,
        load_non_simulated_nodes=False
    ):
        """
        Initialize parameters for neuron report data.

        :param report_id: int, optional: Identifier for the report. Defaults to -1.
        :param value_range: Vector2, optional: Range of values for the mapped data. Defaults to [-80.0, 10.0].
        :param voltage_scaling_range: Vector2, optional: Minimum and maximum scaling factors. Defaults to [1.0, 1.0].
        :param initial_simulation_frame: int, optional: Frame at which simulations start. Defaults to 0.
        :param load_non_simulated_nodes: bool, optional: Whether to load nodes that are not part of the simulation. Defaults to False.
        """
        self.components = {
            'report_id': report_id,
            'value_range': value_range,
            'voltage_scaling_range': voltage_scaling_range,
            'initial_simulation_frame': initial_simulation_frame,
            'load_non_simulated_nodes': load_non_simulated_nodes
        }

    def get_component(self, key):
        """
        Retrieve a specific configuration parameter by key.

        :param key: str: The key corresponding to the desired parameter.
        :return: The value associated with the specified key.
        """
        return self.components.get(key)

    def to_list(self):
        """
        Convert configuration parameters into a list format.

        :return: List containing values of all components in the order they are stored.
        :rtype: list
        """
        return [
            self.components['report_id'],
            self.components['value_range'].x,
            self.components['value_range'].y,
            self.components['voltage_scaling_range'].x,
            self.components['voltage_scaling_range'].y,
            self.components['initial_simulation_frame'],
            int(self.components['load_non_simulated_nodes'])
        ]

    def copy(self):
        """
        Create a copy of the current object.

        :return: NeuronReportParams: A new instance with the same settings.
        """
        return NeuronReportParams(
            report_id=self.components['report_id'],
            value_range=self.components['value_range'],
            voltage_scaling_range=self.components['voltage_scaling_range'],
            initial_simulation_frame=self.components['initial_simulation_frame'],
            load_non_simulated_nodes=self.components['load_non_simulated_nodes']
        )

    def __repr__(self):
        """
        Provide a string representation of the object for debugging and logging purposes.

        :return: A string that represents the object.
        """
        return (f"NeuronReportParams(report_id={self.components['report_id']}, "
                f"value_range={self.components['value_range']}, "
                f"voltage_scaling_range={self.components['voltage_scaling_range']}, "
                f"initial_simulation_frame={self.components['initial_simulation_frame']}, "
                f"load_non_simulated_nodes={self.components['load_non_simulated_nodes']})")
