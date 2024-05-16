# !/usr/bin/env python
"""BioExplorer class"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2024 Blue BrainProject / EPFL
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.


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
        :param position_strength: float, optional: Strength of position randomization. Defaults to 0.0.
        :param rotation_seed: int, optional: Seed for rotation randomization. Defaults to 0.
        :param rotation_strength: float, optional: Strength of rotation randomization. Defaults to 0.0.
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
        """
        Return the official string representation of the object for debugging and logging purposes.
        """
        return f"MolecularSystemAnimationParams({', '.join(f'{k}={v}' for k, v in self.components.items())})"


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
        """
        Return the official string representation of the object for debugging and logging purposes.
        """
        return f"CellAnimationParams({', '.join(f'{k}={v}' for k, v in self.components.items())})"
