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
Module displacement_parameters

This module provides displacement parameters for neurons, astrocytes and vasculature
"""

from .math_utils import Vector2


class NeuronDisplacementParams:
    """
    Parameters used for the sinusoidal Signed Distance Field displacement for neuron morphologies.

    Each neuron component (e.g., soma, section) has associated amplitude and frequency for
    displacement.
    """

    def __init__(
        self,
        soma=Vector2(0.1, 3.0),
        section=Vector2(0.15, 2.0),
        nucleus=Vector2(0.01, 2.0),
        mitochondrion=Vector2(0.2, 100.0),
        myelin_sheath=Vector2(0.1, 2.5),
        spine=Vector2(0.01, 25.0),
    ):
        """
        Initialize displacement parameters for various components of a neuron.

        :param soma: Vector2, optional: Amplitude and frequency for the soma. Defaults to
        [0.1, 3.0].
        :param section: Vector2, optional: Amplitude and frequency for the section. Defaults to
        [0.15, 2.0].
        :param nucleus: Vector2, optional: Amplitude and frequency for the nucleus. Defaults to
        [0.01, 2.0].
        :param mitochondrion: Vector2, optional: Amplitude and frequency for the mitochondrion.
        Defaults to [0.2, 100.0].
        :param myelin_sheath: Vector2, optional: Amplitude and frequency for the myelin sheath.
        Defaults to [0.1, 2.5].
        :param spine: Vector2, optional: Amplitude and frequency for the spine. Defaults to
        [0.01, 25.0].
        """
        self.components = {
            'soma': soma,
            'section': section,
            'nucleus': nucleus,
            'mitochondrion': mitochondrion,
            'myelin_sheath': myelin_sheath,
            'spine': spine
        }

    def get_displacement_params(self, component):
        """
        Retrieve displacement parameters for a specific neuron component.

        :param component: str: Name of the neuron component.
        :return: Vector2: Displacement parameters for the specified component.
        """
        return self.components.get(component, Vector2(0, 0))

    def to_list(self):
        """
        Convert the displacement parameters for all components into a list format.

        :return: list of float: List containing the amplitude and frequency values for all
        components.
        """
        return [value for params in self.components.values() for value in (params.x, params.y)]

    def copy(self):
        """
        Create a copy of the current object.

        :return: NeuronDisplacementParams: A new instance with the same parameters.
        """
        return NeuronDisplacementParams(**self.components)

    def __repr__(self):
        """Provide a string representation of the object for debugging and logging purposes."""
        return f"NeuronDisplacementParams(\
            {', '.join(f'{k}={v}' for k, v in self.components.items())})"


class AstrocyteDisplacementParams:
    """
    Parameters used for the sinusoidal Signed Distance Field displacement function for astrocyte.

    Each astrocyte component (e.g., soma, section) has associated amplitude and frequency for
    displacement.
    """

    def __init__(
        self,
        soma=Vector2(0.05, 0.5),
        section=Vector2(0.5, 5.0),
        nucleus=Vector2(0.01, 2.0),
        mitochondrion=Vector2(0.2, 100.0),
        end_foot=Vector2(0.3, 0.5),
    ):
        """
        Initialize displacement parameters for various components of an astrocyte.

        :param soma: Vector2, optional: Amplitude and frequency for the soma. Defaults to
        [0.05, 0.5].
        :param section: Vector2, optional: Amplitude and frequency for the section. Defaults to
        [0.5, 5.0].
        :param nucleus: Vector2, optional: Amplitude and frequency for the nucleus. Defaults to
        [0.01, 2.0].
        :param mitochondrion: Vector2, optional: Amplitude and frequency for the mitochondrion.
        Defaults to [0.2, 100.0].
        :param end_foot: Vector2, optional: Amplitude and frequency for the end foot. Defaults to
        [0.3, 0.5].
        """
        self.components = {
            'soma': soma,
            'section': section,
            'nucleus': nucleus,
            'mitochondrion': mitochondrion,
            'end_foot': end_foot
        }

    def get_displacement_params(self, component):
        """
        Retrieve displacement parameters for a specific astrocyte component.

        :param component: str: Name of the astrocyte component.
        :return: Vector2: Displacement parameters for the specified component.
        """
        return self.components.get(component, Vector2(0, 0))

    def to_list(self):
        """
        Convert the displacement parameters for all components into a list format.

        :return: list of float: List containing the amplitude and frequency values for all
        components.
        """
        return [value for params in self.components.values() for value in (params.x, params.y)]

    def copy(self):
        """
        Create a copy of the current object.

        :return: AstrocyteDisplacementParams: A new instance with the same parameters.
        """
        return AstrocyteDisplacementParams(**self.components)

    def __repr__(self):
        """Provide a string representation of the object for debugging and logging purposes."""
        return f"AstrocyteDisplacementParams(\
            {', '.join(f'{k}={v}' for k, v in self.components.items())})"


class VasculatureDisplacementParams:
    """Parameters used for the sinusoidal SDF displacement function for vasculature."""

    def __init__(self, segment=Vector2(0.3, 0.5)):
        """
        Initialize displacement parameters for vasculature segments.

        :param segment: Vector2, optional: Amplitude and frequency for the segment. Defaults to
        [0.3, 0.5].
        """
        assert isinstance(segment, Vector2)
        self.segment = segment

    def to_list(self):
        """
        A list containing the values of class members.

        :return: A list containing the values of class members.
        :rtype: list
        """
        return [self.segment.x, self.segment.y]

    def copy(self):
        """
        Copy the current object.

        :return: VasculatureDisplacementParams: A copy of the object.
        """
        return VasculatureDisplacementParams(self.segment)


class SynapseDisplacementParams:
    """
    Parameters used for the sinusoidal Signed Distance Field displacement function for synapse.

    This class defines displacement parameters for the spine component of a synapse.
    """

    def __init__(self, spine=Vector2(0.01, 25.0)):
        """
        Initialize displacement parameters for the spine of a synapse.

        :param spine: Vector2, optional: Amplitude and frequency for the spine. Defaults to
        [0.01, 25.0].
        """
        assert isinstance(spine, Vector2)
        self.components = {'spine': spine}

    def get_displacement_params(self):
        """
        Retrieve displacement parameters for the spine.

        :return: Vector2: Displacement parameters for the spine.
        """
        return self.components['spine']

    def to_list(self):
        """
        Convert the displacement parameters into a list format.

        :return: list of float: List containing the amplitude and frequency values for the spine.
        """
        return [self.components['spine'].x, self.components['spine'].y]

    def copy(self):
        """
        Create a copy of the current object.

        :return: SynapseDisplacementParams: A new instance with the same parameters.
        """
        return SynapseDisplacementParams(spine=self.components['spine'])

    def __repr__(self):
        """Provide a string representation of the object for debugging and logging purposes."""
        return f"SynapseDisplacementParams(spine={self.components['spine']})"
