#!/usr/bin/env

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

"""
Module animation_parameters

This module provides classes to define animation parameters for molecular systems and cells.
These parameters are used to introduce randomness and sinusoidal motion to make animations appear
more realistic.
"""


class MolecularSystemAnimationParams:
    """
    Parameters used to introduce some randomness in the position and orientation of the protein.

    This approach is used to make assemblies appear more realistic and for animation purposes.
    """

    def __init__(
        self,
        seed=0,
        position_seed=0,
        position_strength=0.0,
        rotation_seed=0,
        rotation_strength=0.0,
        morphing_step=0.0,
    ):
        """
        Initialize the animation parameters using a dictionary to store components.

        :param seed: int, optional: Randomization seed for initial setup. Defaults to 0.
        :param position_seed: int, optional: Seed for position randomization. Defaults to 0.
        :param position_strength: float, optional: Strength of position randomization. Defaults to
        0.0.
        :param rotation_seed: int, optional: Seed for rotation randomization. Defaults to 0.
        :param rotation_strength: float, optional: Strength of rotation randomization. Defaults to
        0.0.
        :param morphing_step: float, optional: Step for morphing between shapes. Defaults to 0.0.
        """
        self.components = {
            'seed': seed,
            'position_seed': position_seed,
            'position_strength': position_strength,
            'rotation_seed': rotation_seed,
            'rotation_strength': rotation_strength,
            'morphing_step': morphing_step
        }

    def get_params(self, component):
        """
        Retrieve parameters for a specific component.

        :param component: str: Name of the component.
        :return: float: Parameters for the specified component.
        """
        return self.components.get(component, 0)

    def to_list(self):
        """
        Convert all animation parameters into a list format.

        :return: List of values representing all components.
        :rtype: list
        """
        return list(self.components.values())

    def copy(self):
        """
        Create a copy of the current object, preserving all parameters.

        :return: MolecularSystemAnimationParams: A new instance with duplicated settings.
        """
        return MolecularSystemAnimationParams(**self.components)

    def __repr__(self):
        """Return the string representation of the object for debugging and logging purposes."""
        return f"MolecularSystemAnimationParams(\
            {', '.join(f'{k}={v}' for k, v in self.components.items())})"


class CellAnimationParams:
    """
    Parameters used to introduce some sinusoidal function in a cell structure.

    This class is used to define how cells should be animated using sinusoidal motion.
    """

    def __init__(self, seed=0, offset=0, amplitude=1.0, frequency=1.0):
        """
        Initialize the animation parameters with sinusoidal characteristics.

        :param seed: int, optional: Initial position in the sinusoidal function. Defaults to 0.
        :param offset: int, optional: Offset in the sinusoidal function. Defaults to 0.
        :param amplitude: float, optional: Amplitude of the sinusoidal function.
        :param frequency: float, optional: Frequency of the sinusoidal function.
        """
        self.components = {
            'seed': seed,
            'offset': offset,
            'amplitude': amplitude,
            'frequency': frequency
        }

    def get_params(self, component):
        """
        Retrieve parameters for a specific component.

        :param component: str: Name of the component.
        :return: float: Parameters for the specified component.
        """
        return self.components.get(component, 0)

    def to_list(self):
        """
        Convert all animation parameters into a list format.

        :return: List of values representing all components.
        :rtype: list
        """
        return list(self.components.values())

    def copy(self):
        """
        Create a copy of the current object, preserving all parameters.

        :return: CellAnimationParams: A new instance with duplicated settings.
        """
        return CellAnimationParams(**self.components)

    def __repr__(self):
        """Return the string representation of the object for debugging and logging purposes."""
        return f"CellAnimationParams({', '.join(f'{k}={v}' for k, v in self.components.items())})"
