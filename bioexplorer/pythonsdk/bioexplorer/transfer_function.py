# !/usr/bin/env python
"""Provides a transfer function widget"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2022 Blue BrainProject / EPFL
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

import seaborn as sns
from webcolors import name_to_rgb, hex_to_rgb

from ipywidgets import widgets, Layout, Box, VBox, ColorPicker
from IPython.display import display

# pylint: disable=dangerous-default-value
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument


class TransferFunction:
    """Transfer function widget"""

    def __init__(self, bioexplorer, model_id, filename=None, name='rainbow',
                 size=32, alpha=0.0, value_range=[0, 255],
                 continuous_update=False):
        """
        Initialize the TransferFunction object

        :bioexplorer: BioExplorer client
        :filename: Optional full file name of the transfer function
        :name: Name of the color palette
        :size: Number of control points
        :alpha: Default alpha value
        :value_range: Data range
        :continuous_update: Enable continuous synchronization with the back-end
        """
        self._core = bioexplorer.core_api()
        self._model_id = model_id
        self._palette = list()
        self._alpha_sliders = list()
        self._color_pickers = list()
        self._continuous_update = continuous_update
        self._value_range = value_range
        self._send_updates_to_renderer = True

        if filename is None:
            # Initialize palette from seaborn
            self._palette.clear()
            for color in sns.color_palette(name, size):
                self._palette.append([color[0], color[1], color[2], alpha])
        else:
            # Load palette from file
            self._load(filename)

        # Create control and assign palette
        self._create_controls()
        self._update_controls()
        self._callback()

    def _load(self, filename):
        """
        Load transfer function from file

        :filename: Full file name of the transfer function
        """
        # Clear controls
        self._alpha_sliders.clear()
        self._color_pickers.clear()

        # Read colormap file
        lines = tuple(open(filename, 'r'))
        self._palette.clear()
        for line in lines:
            words = line.split()
            if len(words) == 4:
                r = float(words[0])
                g = float(words[1])
                b = float(words[2])
                a = float(words[3])
                color = [r, g, b, a]
                self._palette.append(color)

    def save(self, filename):
        """
        Save transfer function to file

        :filename: Full file name of the transfer function
        """
        with open(filename, 'w') as f:
            f.write(str(len(self._palette)) + '\n')
            for color in self._palette:
                f.write(str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' ' +
                        str(color[3]) + '\n')
            f.close()

    def set_palette(self, name):
        """
        Set the Seaborn color palette of the transfer function

        :name: Name of the Seaborn color palette
        """
        size = len(self._palette)
        newPalette = sns.color_palette(name, size)
        for i in range(size):
            color = newPalette[i]
            self._palette[i] = [color[0], color[1], color[2], self._palette[i][3]]
        self._update_controls()
        self._callback()

    def set_range(self, value_range):
        """
        Set the transfer function value_range

        :name: Name of the Seaborn color palette
        """
        self._value_range = value_range
        self._callback()

    def _html_color(self, index):
        """
        Get HTML color from palette

        :index: Index of color in the Seaborn color palette
        """
        color = self._palette[index]
        color_as_string = '#' \
                          '%02x' % (int)(color[0] * 255) + \
                          '%02x' % (int)(color[1] * 255) + \
                          '%02x' % (int)(color[2] * 255)
        return color_as_string

    def _update_colormap(self, change):
        """
        Update color map

        :change: Unused
        """
        self._callback()

    def _update_colorpicker(self, change):
        """
        Update color picker

        :change: Unused
        """
        for i in range(len(self._palette)):
            self._alpha_sliders[i].style.handle_color = self._color_pickers[i].value
        self._callback()

    def _create_controls(self):
        """Create widget controls"""
        self.send_updates_to_renderer = False
        # Layout
        alpha_slider_item_layout = Layout(
            overflow_x='hidden', height='180px', max_width='20px')
        color_picker_item_layout = Layout(
            overflow_x='hidden', height='20px', max_width='20px')
        box_layout = Layout(display='inline-flex')

        # Sliders
        self._alpha_sliders = [widgets.IntSlider(
            continuous_update=self._continuous_update,
            layout=alpha_slider_item_layout,
            description=str(i),
            orientation='vertical',
            readout=True,
            value=self._palette[i][3] * 256, min=0, max=255, step=1
        ) for i in range(len(self._palette))]

        # Color pickers
        self._color_pickers = [
            ColorPicker(
                layout=color_picker_item_layout,
                concise=True,
                disabled=False) for i in range(len(self._palette))
        ]
        # Display controls
        color_box = Box(children=self._color_pickers)
        alpha_box = Box(children=self._alpha_sliders)
        box = VBox([color_box, alpha_box], layout=box_layout)

        # Attach observers
        for i in range(len(self._palette)):
            self._alpha_sliders[i].observe(self._update_colormap, names='value')
            self._color_pickers[i].observe(self._update_colorpicker, names='value')
        display(box)
        self._send_updates_to_renderer = True

    def _update_controls(self):
        """Udpate widget controls"""
        self._send_updates_to_renderer = False
        for i in range(len(self._palette)):
            color = self._html_color(i)
            self._alpha_sliders[i].style.handle_color = color
            self._color_pickers[i].value = color
        self._send_updates_to_renderer = True

    def _callback(self):
        """Call back method for widget changes"""
        if not self._send_updates_to_renderer:
            return

        for i in range(len(self._palette)):
            try:
                color = name_to_rgb(self._color_pickers[i].value)
            except ValueError:
                color = hex_to_rgb(self._color_pickers[i].value)

            c = [
                float(color.red) / 255.0,
                float(color.green) / 255.0,
                float(color.blue) / 255.0,
                float(self._alpha_sliders[i].value) / 255.0
            ]
            self._palette[i] = c
        self._update_palette()

    @staticmethod
    def _hex_to_rgb(value):
        """Concert hex value into RGB values"""
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def _update_palette(self):
        """Update color palette"""
        intensity = 1
        btf = self._core.get_model_transfer_function(id=self._model_id)
        colors = list()
        points = list()
        nb_points = len(self._alpha_sliders)
        step = 1.0 / float(nb_points - 1)
        for i in range(nb_points):
            color = self._hex_to_rgb(self._color_pickers[i].value)
            colors.append([
                intensity * float(color[0]) / 256.0,
                intensity * float(color[1]) / 256.0,
                intensity * float(color[2]) / 256.0])
            points.append([i * step, self._alpha_sliders[i].value / 255.0])

        btf['colormap']['name'] = 'TransferFunctionEditor'
        btf['colormap']['colors'] = colors
        btf['opacity_curve'] = points
        btf['range'] = [self._value_range[0], self._value_range[1]]
        self._core.set_model_transfer_function(id=self._model_id, transfer_function=btf)
