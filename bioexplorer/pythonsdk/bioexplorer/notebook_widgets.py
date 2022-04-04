#!/usr/bin/env python
"""BioExplorer widgets"""

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

import math
import random
import threading
import time
import glob
import os
import io
from ipywidgets import FloatSlider, Select, HBox, VBox, Layout, Button, SelectMultiple, \
    Checkbox, IntRangeSlider, ColorPicker, IntSlider, Label, Text, Image
from IPython.display import display
import matplotlib
import seaborn as sns
from stringcase import pascalcase
from PIL import ImageDraw
from .bio_explorer import Vector3


# pylint: disable=unused-argument
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements

COLOR_MAPS = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
    'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
    'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
    'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
    'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
    'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
    'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr',
    'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
    'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cool', 'cool_r',
    'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
    'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
    'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
    'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
    'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r',
    'inferno', 'inferno_r', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r',
    'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma',
    'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic',
    'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
    'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
    'terrain', 'terrain_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
]

SHADING_MODES = [
    'none', 'basic', 'diffuse', 'electron', 'cartoon', 'electron_transparency', 'perlin',
    'diffuse_transparency', 'checker', 'goodsell'
]

CHAMELEON_MODES = [
    'none', 'emitter', 'receiver'
]

DEFAULT_GRID_LAYOUT = Layout(border='1px solid black', margin='5px', padding='5px')
DEFAULT_LAYOUT = Layout(width='50%', height='24px', display='flex', flex_flow='row')
STYLE = {'description_width': 'initial', 'handle_color': 'gray'}


class Widgets:
    """Set of notebook widgets for the BioExplorer"""

    def __init__(self, bioexplorer):
        """Initialize with a reference to BioExplorer and underlying Brayns API"""
        self._be = bioexplorer
        self._client = bioexplorer.core_api()

    def display_focal_distance(self, with_preview=False):
        """Display visual controls for setting camera focal distance"""
        x_slider = FloatSlider(description='X', min=0, max=1, value=0.5)
        y_slider = FloatSlider(description='Y', min=0, max=1, value=0.5)
        a_slider = FloatSlider(description='Aperture', min=0, max=1, value=0)
        f_slider = FloatSlider(description='Focus radius', min=0, max=1, value=0.01)
        d_slider = FloatSlider(description='Focus distance', min=0,
                               max=10000, value=0, disabled=True)
        f_button = Button(description='Refresh')
        f_target = Button(description='Target')

        class Updated:
            """Class object embedding communication with remote server"""

            def __init__(self, client, with_preview):
                self._client = client
                self._widget_value = None
                self._x = 0.5
                self._y = 0.5
                self._aperture = 0.0
                self._focus_radius = 0.01
                self._focus_distance = 0.0
                self._nb_focus_points = 20
                self._snapshot = None
                self._with_preview = with_preview

            def _update_camera(self):
                self._focus_distance = 1e6
                for _ in range(self._nb_focus_points):
                    self._focus_distance = min(self._focus_distance, self._get_focal_distance(
                        (self._x + random.random() * self._focus_radius,
                         self._y + random.random() * self._focus_radius)))

                params = self._client.BioExplorerPerspectiveCameraParams()
                params.focus_distance = self._focus_distance
                params.aperture_radius = self._aperture
                params.enable_clipping_planes = True
                d_slider.value = self._focus_distance
                self._client.set_camera_params(params)

                if self._with_preview:
                    self._get_preview(False)

            def update(self):
                """Update all settings of the camera"""
                self._update_camera()

                if self._with_preview:
                    self._get_preview(True)

            def update_target(self):
                """Update camera target"""
                inspection = self._client.inspect([self._x, self._y])
                if inspection['hit']:
                    position = inspection['position']
                    self._client.set_camera(target=position)
                    self._client.set_renderer()

            def update_focus_radius(self, val_dict) -> None:
                """Update camera focus radius"""
                self._widget_value = val_dict['new']
                self._focus_radius = self._widget_value
                self._update_camera()

            def update_aperture(self, val_dict) -> None:
                """Update camera aperture"""
                self._widget_value = val_dict['new']
                self._aperture = self._widget_value
                self._update_camera()

            def update_x(self, val_dict) -> None:
                """Update camera normalized horizontal focus location"""
                self._widget_value = val_dict['new']
                self._x = self._widget_value
                self._update_camera()

            def update_y(self, val_dict) -> None:
                """Update camera normalized vertical focus location"""
                self._widget_value = val_dict['new']
                self._y = self._widget_value
                self._update_camera()

            def _get_focal_distance(self, coordinates=(0.5, 0.5)):
                """
                Return the focal distance for the specified normalized coordinates in the image

                :param list coordinates: Coordinates in the image
                :return: The focal distance
                :rtype: float
                """
                target = self._client.inspect(array=coordinates)['position']
                origin = self._client.camera.position.data
                vector = [0, 0, 0]
                for k in range(3):
                    vector[k] = float(target[k]) - float(origin[k])
                return math.sqrt(
                    vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

            def _get_preview(self, update_image):
                viewport = self._client.get_application_parameters()['viewport']
                ratio = viewport[1] / viewport[0]
                size = [256, int(256.0 * ratio)]
                if update_image:
                    self._snapshot = self._client.image(size=size, samples_per_pixel=4)

                if self._snapshot is None:
                    return

                x = self._x * size[0]
                y = (1.0 - self._y) * size[1]
                rx = self._focus_radius * size[0]
                ry = self._focus_radius * size[1]
                byte_io = io.BytesIO()
                snapshot = self._snapshot.copy()
                draw = ImageDraw.Draw(snapshot)
                draw.ellipse((x - rx, y - ry, x + rx, y + ry))
                draw.ellipse((x - 5, y - 5, x + 5, y + 5))
                snapshot.save(byte_io, 'png')
                preview.width = size[0]
                preview.height = size[1]
                preview.value = byte_io.getvalue()

        update_class = Updated(self._client, with_preview)

        def update_x(value):
            update_class.update_x(value)

        def update_y(value):
            update_class.update_y(value)

        def update_aperture(value):
            update_class.update_aperture(value)

        def update_focus_radius(value):
            update_class.update_focus_radius(value)

        def update_button(_):
            update_class.update()

        def update_target(_):
            update_class.update_target()

        x_slider.observe(update_x, 'value')
        y_slider.observe(update_y, 'value')
        a_slider.observe(update_aperture, 'value')
        f_slider.observe(update_focus_radius, 'value')
        f_button.on_click(update_button)
        f_target.on_click(update_target)

        position_box = VBox([x_slider, y_slider, f_button, f_target])
        parameters_box = VBox([a_slider, f_slider, d_slider])
        horizontal_box = HBox([position_box, parameters_box], layout=DEFAULT_GRID_LAYOUT)
        display(horizontal_box)

        if with_preview:
            byte_io = io.BytesIO()
            preview = Image(value=byte_io.getvalue(), format='png', width=1, height=1)
            display(preview)

    def display_palette_for_models(self):
        """Display visual controls for color palettes applied to models"""
        def set_colormap(model_id, colormap_name, shading_mode):
            material_ids = self._be.get_material_ids(model_id)['ids']
            nb_materials = len(material_ids)

            palette = sns.color_palette(colormap_name, nb_materials)
            self._be.set_materials_from_palette(
                model_ids=[model_id], material_ids=material_ids, palette=palette,
                specular_exponent=specular_exponent_slider.value, shading_mode=shading_mode,
                opacity=opacity_slider.value,
                refraction_index=refraction_index_slider.value,
                reflection_index=reflection_index_slider.value,
                glossiness=glossiness_slider.value,
                user_parameter=user_param_slider.value,
                emission=emission_slider.value,
                chameleon_mode=chameleon_combobox.index
            )
            self._client.set_renderer(accumulation=True)

        # Models
        model_names = list()
        for model in self._client.scene.models:
            model_names.append(model['name'])
        model_combobox = Select(options=model_names, description='Models:', disabled=False)

        # Shading modes
        shading_combobox = Select(options=SHADING_MODES, description='Shading:', disabled=False)

        # Colors
        palette_combobox = Select(options=COLOR_MAPS, description='Palette:', disabled=False)

        # Chameleon modes
        chameleon_combobox = Select(options=CHAMELEON_MODES,
                                    description='Chameleon:', disabled=False)

        # Events
        def update_materials_from_palette(value):
            """Update materials when palette is modified"""
            set_colormap(self._client.scene.models[model_combobox.index]['id'], value['new'],
                         shading_combobox.index)

        def update_materials_from_shading_modes(_):
            """Update materials when shading is modified"""
            set_colormap(self._client.scene.models[model_combobox.index]['id'],
                         palette_combobox.value, shading_combobox.index)

        def update_materials_from_chameleon_modes(_):
            """Update materials when chameleon mode is modified"""
            set_colormap(self._client.scene.models[model_combobox.index]['id'],
                         palette_combobox.value, shading_combobox.index)

        shading_combobox.observe(update_materials_from_shading_modes, 'value')
        palette_combobox.observe(update_materials_from_palette, 'value')
        chameleon_combobox.observe(update_materials_from_chameleon_modes, 'value')

        horizontal_box_list = HBox([model_combobox, shading_combobox, palette_combobox])

        opacity_slider = FloatSlider(description='Opacity', min=0, max=1, value=1)
        opacity_slider.observe(update_materials_from_shading_modes)
        refraction_index_slider = FloatSlider(description='Refraction', min=1, max=5, value=1)
        refraction_index_slider.observe(update_materials_from_shading_modes)
        reflection_index_slider = FloatSlider(description='Reflection', min=0, max=1, value=0)
        reflection_index_slider.observe(update_materials_from_shading_modes)
        glossiness_slider = FloatSlider(description='Glossiness', min=0, max=1, value=1)
        glossiness_slider.observe(update_materials_from_shading_modes)
        specular_exponent_slider = FloatSlider(description='Specular exponent', min=1, max=100,
                                               value=1)
        specular_exponent_slider.observe(update_materials_from_shading_modes)
        user_param_slider = FloatSlider(description='User param', min=0, max=100, value=1)
        user_param_slider.observe(update_materials_from_shading_modes)
        emission_slider = FloatSlider(description='Emission', min=0, max=100, value=0)
        emission_slider.observe(update_materials_from_shading_modes)

        cast_simulation_checkbox = Checkbox(description='Simulation', value=False)
        cast_simulation_checkbox.observe(update_materials_from_shading_modes)

        horizontal_box_detail1 = HBox([opacity_slider, refraction_index_slider,
                                       reflection_index_slider])
        horizontal_box_detail2 = HBox(
            [glossiness_slider, specular_exponent_slider, user_param_slider])
        horizontal_box_detail3 = HBox(
            [emission_slider, cast_simulation_checkbox, chameleon_combobox])
        vertical_box = VBox([horizontal_box_list, horizontal_box_detail1, horizontal_box_detail2,
                             horizontal_box_detail3], layout=DEFAULT_GRID_LAYOUT)
        display(vertical_box)

    def display_model_visibility(self):
        """Display visual controls for setting visibility of models"""
        model_names = list()
        for model in self._client.scene.models:
            model_names.append(model['name'])

        model_select = SelectMultiple(options=model_names, disabled=False)

        lbl_models = Label(value='Models:')
        show_btn = Button(description='Show')
        hide_btn = Button(description='Hide')
        lbl_aabb = Label(value='Bounds:')
        show_aabb_btn = Button(description='Show')
        hide_aabb_btn = Button(description='Hide')
        lbl_camera = Label(value='Camera:')
        adjust_camera_btn = Button(description='Adjust')
        vbox_visible = VBox([lbl_models, show_btn, hide_btn])
        vbox_aabb = VBox([lbl_aabb, show_aabb_btn, hide_aabb_btn])
        vbox_camera = VBox([lbl_camera, adjust_camera_btn])
        hbox_params = HBox([vbox_visible, vbox_aabb, vbox_camera])

        def update_models(visible):
            for model_id in model_select.index:
                self._client.update_model(
                    id=self._client.scene.models[model_id]['id'],
                    visible=visible)

        def show_models(event):
            update_models(True)

        def hide_models(event):
            update_models(False)

        def update_aabbs(visible):
            for model_id in model_select.index:
                self._client.update_model(
                    id=self._client.scene.models[model_id]['id'],
                    bounding_box=visible)

        def show_aabbs(event):
            update_aabbs(True)

        def hide_aabbs(event):
            update_aabbs(False)

        def adjust_camera(event):
            size_min_aabb = [1e6, 1e6, 1e6]
            size_max_aabb = [-1e6, -1e6, -1e6]
            for model_id in model_select.index:
                model = self._client.scene.models[model_id]
                bounds = model['bounds']
                min_aabb = bounds['min']
                max_aabb = bounds['max']
                half_size_aabb = [0, 0, 0]
                center = [0, 0, 0]
                for k in range(3):
                    half_size_aabb[k] = (max_aabb[k] - min_aabb[k]) / 2.0
                    center[k] = center[k] + (min_aabb[k] + max_aabb[k]) / 2.0

                for k in range(3):
                    size_min_aabb[k] = min(size_min_aabb[k], center[k] - half_size_aabb[k])
                    size_max_aabb[k] = max(size_max_aabb[k], center[k] + half_size_aabb[k])

            center_aabb = [0, 0, 0]
            diag = 0
            for k in range(3):
                diag = max(diag, size_max_aabb[k] - size_min_aabb[k])
                center_aabb[k] = (size_max_aabb[k] + size_min_aabb[k]) / 2.0

            origin = [0, 0, 0]
            for k in range(3):
                origin[k] = center_aabb[k]
                if k == 2:
                    origin[k] = origin[k] + diag * 1.5

            self._client.set_camera(position=origin, orientation=[0, 0, 0, 1], target=center_aabb)
            # Refresh UI
            self._client.set_renderer()

        show_btn.on_click(show_models)
        hide_btn.on_click(hide_models)
        show_aabb_btn.on_click(show_aabbs)
        hide_aabb_btn.on_click(hide_aabbs)
        adjust_camera_btn.on_click(adjust_camera)

        hbox = HBox([model_select, hbox_params], layout=DEFAULT_GRID_LAYOUT)
        display(hbox)

    def display_model_focus(self, max_number_of_instances=1e6):
        """Display visual controls for setting visibility of models"""
        directions = dict()
        directions['front'] = Vector3(0, 0, -1)
        directions['back'] = Vector3(0, 0, 1)
        directions['top'] = Vector3(0, -1, 0)
        directions['bottom'] = Vector3(0, 1, 0)
        directions['right'] = Vector3(-1, 0, 0)
        directions['left'] = Vector3(1, 0, 0)

        # Model names and ids
        model_names = list()
        model_ids = self._be.get_model_ids()['ids']
        for model_id in model_ids:
            model_name = self._be.get_model_name(model_id)['name']
            if self._be.NAME_MEMBRANE not in model_name:
                model_names.append([model_name, model_id])
        model_select = Select(
            description='Model',
            options=model_names)

        # Instance Ids
        instance_select = Select(
            description='Instance',
            options=list())

        # Directions
        direction_select = Select(
            description='Direction',
            options=directions)

        # Distance to instance
        distance_slider = FloatSlider(
            description='Distance', value=25.0, min=-1e3, max=1e3)

        # Focus button
        focus_btn = Button(description='Focus')
        hbox_1 = HBox([model_select, instance_select, direction_select])
        hbox_2 = HBox([distance_slider, focus_btn])
        vbox = VBox([hbox_1, hbox_2])

        def update_instances(value):
            model_id = int(model_select.options[model_select.index][1])
            ids = self._be.get_model_instances(model_id, max_number_of_instances)['ids']
            instance_select.options = ids

        def focus_on_instance(event):
            model_id = int(model_select.options[model_select.index][1])
            instance_id = int(instance_select.options[instance_select.index])
            self._be.set_focus_on(
                model_id=model_id, instance_id=instance_id,
                distance=distance_slider.value,
                direction=direction_select.value,
                max_number_of_instances=max_number_of_instances
            )
            self._client.set_renderer()

        model_select.observe(update_instances, 'value')
        focus_btn.on_click(focus_on_instance)
        update_instances(0)

        display(vbox)

    def __display_advanced_settings(self, object_type, threaded):
        """Display visual controls for camera or renderer advanced settings"""
        class Updated:
            """Inner class that insures communication with the remote server"""

            def __init__(self, client, object_type):
                self._object_type = object_type
                self._widgets_list = dict()
                self._native_params = None
                self._client = client
                class_name = None
                if self._object_type == 'camera':
                    class_name = pascalcase(self._client.get_camera()['current'])
                    class_name += 'CameraParams'
                elif self._object_type == 'renderer':
                    class_name = pascalcase(self._client.get_renderer()['current'])
                    class_name += 'RendererParams'

                # Read params from Brayns
                self._native_params = self._get_params()

                # Read params from class definition
                class_ = getattr(self._client, class_name)
                self._param_descriptors = class_()

                # Only keep params that belong to the class definition
                params_to_call = getattr(self._client, class_name)
                self._params = params_to_call()
                if self._native_params is not None:
                    for param in self._native_params:
                        if param in self._params:
                            self._params[param] = self._native_params[param]

                self._create_widgets()

            def get_widgets(self):
                """Returns the list of widgets for that visual control"""
                return self._widgets_list

            def thread_run(self):
                """Runs background thread to update renderer settings in the visual controls"""
                while True:
                    params = self._get_params()
                    for widget in self._widgets_list:
                        self._widgets_list[widget].value = params[widget]
                    time.sleep(1)

            def _get_params(self):
                """Gets camera or renderer settings from remote server"""
                if self._object_type == 'camera':
                    return self._client.get_camera_params()
                if self._object_type == 'renderer':
                    return self._client.get_renderer_params()
                return None

            def _update_params(self, _):
                """Update remote camera or renderer params"""
                for widget in self._widgets_list:
                    setattr(self._params, widget, self._widgets_list[widget].value)

                if self._object_type == 'camera':
                    self._client.set_camera_params(self._params)
                if self._object_type == 'renderer':
                    self._client.set_renderer_params(self._params)

            @ staticmethod
            def _get_value(props, key, default_value):
                """Return value of a property"""
                try:
                    return props[key]
                except KeyError as _:
                    return default_value

            def _create_widgets(self):
                """Create widget in the current visual control"""
                for param in self._param_descriptors:
                    value = self._native_params[param]
                    parameter = self._param_descriptors[param]
                    props = parameter.__propinfo__['__literal__']
                    description = props['title']

                    if isinstance(value, float):
                        minimum = self._get_value(props, 'minimum', 0)
                        maximum = self._get_value(props, 'maximum', 1e4)
                        float_slider = FloatSlider(
                            description=description, min=minimum, max=maximum, value=value,
                            STYLE=STYLE, layout=DEFAULT_LAYOUT)
                        float_slider.observe(self._update_params, 'value')
                        self._widgets_list[param] = float_slider
                    if isinstance(value, int):
                        minimum = self._get_value(props, 'minimum', 0)
                        maximum = self._get_value(props, 'maximum', 1e4)
                        int_slider = IntSlider(
                            description=description, min=minimum, max=maximum, value=value,
                            STYLE=STYLE, layout=DEFAULT_LAYOUT)
                        int_slider.observe(self._update_params, 'value')
                        self._widgets_list[param] = int_slider
                    if isinstance(value, bool):
                        check_box = Checkbox(
                            description=description, value=bool(value), STYLE=STYLE,
                            layout=DEFAULT_LAYOUT)
                        check_box.observe(self._update_params, 'value')
                        self._widgets_list[param] = check_box
                    if isinstance(value, str):
                        text_box = Text(description=description, value=value, STYLE=STYLE,
                                        layout=DEFAULT_LAYOUT)
                        text_box.observe(self._update_params, 'value')
                        self._widgets_list[param] = text_box

        update_class = Updated(self._client, object_type)
        if threaded:
            the_thread = threading.Thread(target=update_class.thread_run)
            the_thread.start()

        widgets_list = update_class.get_widgets()
        vboxes = list()
        nb_widgets = len(widgets_list)
        nb_columns = 2
        for i in range(0, nb_widgets, nb_columns):
            box_widgets = list()

            for widget in list(widgets_list)[i:min(i + nb_columns, nb_widgets)]:
                box_widgets.append(widgets_list[widget])

            vboxes.append(HBox(box_widgets))

        hbox = VBox(vboxes, layout=DEFAULT_GRID_LAYOUT)
        display(hbox)

    def display_advanced_rendering_settings(self, is_threaded=True):
        """Display visual controls for renderer advanced settings"""
        self.__display_advanced_settings('renderer', is_threaded)

    def display_advanced_camera_settings(self, is_threaded=True):
        """Display visual controls for camera advanced settings"""
        self.__display_advanced_settings('camera', is_threaded)

    def display_rendering_settings(self):
        """Display visual controls for renderer settings"""
        def update_params(_):
            """Update renderer params"""
            self._client.set_renderer(
                accumulation=checkbox_accumulation.value,
                background_color=matplotlib.colors.to_rgb(colorpicker_background.value),
                head_light=checkbox_head_light.value,
                samples_per_pixel=slider_samples_per_pixel.value,
                max_accum_frames=slider_max_accum_frames.value,
                subsampling=slider_sub_sampling.value
            )

        params = self._client.get_renderer()
        slider_samples_per_pixel = IntSlider(description='Samples per pixel', min=1, max=1024,
                                             value=params['samples_per_pixel'])
        slider_samples_per_pixel.observe(update_params)
        slider_max_accum_frames = IntSlider(description='Max accumulation frames', min=1, max=1024,
                                            value=params['max_accum_frames'])
        slider_max_accum_frames.observe(update_params, 'value')
        slider_sub_sampling = IntSlider(description='Sub-sampling', min=1, max=16,
                                        value=params['subsampling'])
        slider_sub_sampling.observe(update_params, 'value')
        colorpicker_background = ColorPicker(description='Background color',
                                             value=matplotlib.colors.to_hex(
                                                 params['background_color']))
        colorpicker_background.observe(update_params, 'value')
        checkbox_head_light = Checkbox(description='Head light', value=params['head_light'])
        checkbox_head_light.observe(update_params, 'value')
        checkbox_accumulation = Checkbox(description='Accumulation', value=params['accumulation'])
        checkbox_accumulation.observe(update_params, 'value')
        hbox_1 = HBox([slider_samples_per_pixel, slider_sub_sampling, slider_max_accum_frames])
        hbox_2 = HBox([colorpicker_background, checkbox_head_light, checkbox_accumulation])
        display(VBox([hbox_1, hbox_2], layout=DEFAULT_GRID_LAYOUT))

    def display_environment_maps(self, folder):
        """Display visual controls for setting environment map"""
        supported_extensions = ['jpg', 'jpeg', 'png']
        hdri_files = list()
        for extension in supported_extensions:
            hdri_files = hdri_files + glob.glob(folder + '/*.' + extension)
        hdri_files.sort()
        base_names = list()
        for hdri_file in hdri_files:
            base_names.append(os.path.basename(hdri_file))

        def update_envmap(value):
            filename = folder + '/' + value['new']
            self._client.set_environment_map(filename)

        cb_names = Select(description='Maps', options=base_names)
        cb_names.observe(update_envmap, 'value')
        display(cb_names)

    def show_amino_acid_on_protein(self, assembly_name, name, sequence_id=0, palette_name='Set1',
                                   palette_size=2):
        """
        Display visual controls for showing amino acid sequence on a protein of the scene

        :param: assembly_name: Name of the assembly containing the protein
        :param: name: Name of the protein
        :param: sequence_id: ID of the protein sequence
        :param: palette_name: Name of the color palette
        :param: palette_size: Size of the color palette
        """
        sequences = self._be.get_protein_amino_acid_sequences(assembly_name, name)
        if sequence_id >= len(sequences):
            raise RuntimeError('Invalid sequence Id')
        sequence_as_list = sequences[0].split(',')

        value_range = [int(sequence_as_list[0]), int(sequence_as_list[1])]
        irs = IntRangeSlider(value=[value_range[0], value_range[1]],
                             min=value_range[0], max=value_range[1])
        lbl = Label(value="AA sequence")

        def update_slider(value):
            self._be.set_protein_amino_acid_sequence_as_range(assembly_name, name, value['new'])
            self._be.set_protein_color_scheme(
                assembly_name, name, self._be.COLOR_SCHEME_AMINO_ACID_SEQUENCE, palette_name,
                palette_size)
            lbl.value = sequence_as_list[2][value['new'][0] -
                                            value_range[0]:value['new'][1] - value_range[0]]

        irs.observe(update_slider, 'value')
        display(irs)
        display(lbl)
