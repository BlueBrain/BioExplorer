# !/usr/bin/env python
"""Provides a class that ease the definition of smoothed camera paths"""

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


import copy
import time
from ipywidgets import IntSlider, IntProgress
from IPython.display import display
from .bio_explorer import BioExplorer
from .version import VERSION as __version__

# pylint: disable=no-member
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


class MovieMaker:
    """Movie maker"""

    PLUGIN_API_PREFIX = 'mm-'
    FRAME_BUFFER_MODE_COLOR = 0
    FRAME_BUFFER_MODE_DEPTH = 1

    def __init__(self, bioexplorer):
        """
        Initialize the MovieMaker object

        :bioexplorer: BioExplorer client
        """
        assert isinstance(bioexplorer, BioExplorer)
        self._client = bioexplorer.core_api()
        self._smoothed_key_frames = list()

    @staticmethod
    def version():
        """
        Get the version of the SDK

        :return: The version of the SDK
        :rtype: string
        """
        return __version__

    def build_camera_path(self, control_points, nb_steps_between_control_points, smoothing_size=1):
        """
        Build a camera path from control points

        :control_points: List of control points
        :nb_steps_between_control_points: Number of steps between two control points
        :smoothing_size: Number of steps to be considered for the smoothing of the path
        """
        origins = list()
        directions = list()
        ups = list()
        aperture_radii = list()
        focus_distances = list()
        self._smoothed_key_frames.clear()

        for s in range(len(control_points) - 1):

            p0 = control_points[s]
            p1 = control_points[s + 1]

            for i in range(nb_steps_between_control_points):
                origin = [0, 0, 0]
                direction = [0, 0, 0]
                up = [0, 0, 0]

                t_origin = [0, 0, 0]
                t_direction = [0, 0, 0]
                t_up = [0, 0, 0]
                for k in range(3):
                    t_origin[k] = (p1['origin'][k] - p0['origin'][k]) / \
                        float(nb_steps_between_control_points)
                    t_direction[k] = (p1['direction'][k] - p0['direction'][k]) / \
                        float(nb_steps_between_control_points)
                    t_up[k] = (p1['up'][k] - p0['up'][k]) / float(nb_steps_between_control_points)

                    origin[k] = p0['origin'][k] + t_origin[k] * float(i)
                    direction[k] = p0['direction'][k] + t_direction[k] * float(i)
                    up[k] = p0['up'][k] + t_up[k] * float(i)

                t_aperture_radius = (p1['apertureRadius'] - p0['apertureRadius']) / float(
                    nb_steps_between_control_points)
                aperture_radius = p0['apertureRadius'] + t_aperture_radius * float(i)

                t_focus_distance = (p1['focusDistance'] - p0['focusDistance']) / \
                    float(nb_steps_between_control_points)
                focus_distance = p0['focusDistance'] + t_focus_distance * float(i)

                origins.append(origin)
                directions.append(direction)
                ups.append(up)
                aperture_radii.append(aperture_radius)
                focus_distances.append(focus_distance)

        nb_frames = len(origins)
        for i in range(nb_frames):
            o = [0, 0, 0]
            d = [0, 0, 0]
            u = [0, 0, 0]
            aperture_radius = 0.0
            focus_distance = 0.0
            for j in range(int(smoothing_size)):
                index = int(max(0, min(i + j - smoothing_size / 2, nb_frames - 1)))
                for k in range(3):
                    o[k] = o[k] + origins[index][k]
                    d[k] = d[k] + directions[index][k]
                    u[k] = u[k] + ups[index][k]
                aperture_radius = aperture_radius + aperture_radii[index]
                focus_distance = focus_distance + focus_distances[index]
            self._smoothed_key_frames.append(
                [
                    (o[0] / smoothing_size, o[1] / smoothing_size, o[2] / smoothing_size),
                    (d[0] / smoothing_size, d[1] / smoothing_size, d[2] / smoothing_size),
                    (u[0] / smoothing_size, u[1] / smoothing_size, u[2] / smoothing_size),
                    aperture_radius / smoothing_size, focus_distance / smoothing_size
                ])
        last = control_points[len(control_points) - 1]
        self._smoothed_key_frames.append(
            (last['origin'], last['direction'], last['up'], last['apertureRadius'],
             last['focusDistance']))

    def get_nb_frames(self):
        """
        Get the number of smoothed frames

        :return: The number of smoothed frames
        :rtype: integer
        """
        return len(self._smoothed_key_frames)

    def get_key_frame(self, frame):
        """
        Get the smoothed camera information for the given frame

        :frame: Frame number
        :return: The smoothed camera information for the given frame
        :rtype: list
        :raises KeyError: if the frame is out of range
        """
        if frame < len(self._smoothed_key_frames):
            return self._smoothed_key_frames[frame]
        raise KeyError

    def set_camera(self, origin, direction, up):
        """
        Set the camera using origin, direction and up vectors

        :origin: Origin of the camera
        :direction: Direction in which the camera is looking
        :up: Up vector
        :return: Result of the request submission
        :rtype: Response
        """
        params = dict()
        params['origin'] = origin
        params['direction'] = direction
        params['up'] = up
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + 'set-odu-camera', params)

    def get_camera(self):
        """
        Get the origin, direction and up vector of the camera

        :return: A JSon representation of the origin, direction and up vectors
        :rtype: Response
        """
        return self._client.rockets_client.request(self.PLUGIN_API_PREFIX + 'get-odu-camera')

    def export_frames(self, size, path, base_name, image_format='png',
                      animation_frames=list(), quality=100, samples_per_pixel=1, start_frame=0,
                      end_frame=0, interpupillary_distance=0.0, export_intermediate_frames=False,
                      frame_buffer_mode=FRAME_BUFFER_MODE_COLOR):
        """
        Exports frames to disk. Frames are named using a 6 digit representation of the frame number

        :path: Folder into which frames are exported
        :image_format: Image format (the ones supported par Brayns: PNG, JPEG, etc)
        :quality: Quality of the exported image (Between 0 and 100)
        :samples_per_pixel: Number of samples per pixels
        :start_frame: Optional value if the rendering should start at a specific frame.
        :end_frame: Optional value if the rendering should end at a specific frame.
        :export_intermediate_frames: Exports intermediate frames (for every sample per pixel)
        :interpupillary_distance: Distance between pupils. Stereo mode is activated if different
        from zero. This is used to resume the rendering of a previously canceled sequence)
        :return: Result of the request submission
        :rtype: Response
        """
        nb_frames = self.get_nb_frames()
        if end_frame == 0:
            end_frame = nb_frames

        assert isinstance(size, list)
        assert len(size) == 2
        if len(animation_frames) != 0:
            assert len(animation_frames) == nb_frames
        assert start_frame <= end_frame
        assert end_frame <= nb_frames

        self._client.set_application_parameters(viewport=size)
        self._client.set_renderer(
            accumulation=True, samples_per_pixel=1, max_accum_frames=samples_per_pixel + 1,
            subsampling=1)

        camera_definitions = list()
        for i in range(start_frame, end_frame):
            camera_definitions.append(self.get_key_frame(i))

        params = dict()
        params['path'] = path
        params['baseName'] = base_name
        params['format'] = image_format
        params['quality'] = quality
        params['spp'] = samples_per_pixel
        params['startFrame'] = start_frame
        params['endFrame'] = end_frame
        params['exportIntermediateFrames'] = export_intermediate_frames
        params['animationInformation'] = animation_frames
        params['frameBufferMode'] = frame_buffer_mode
        values = list()
        for camera_definition in camera_definitions:
            # Origin
            for i in range(3):
                values.append(camera_definition[0][i])
            # Direction
            for i in range(3):
                values.append(camera_definition[1][i])
            # Up
            for i in range(3):
                values.append(camera_definition[2][i])
            # Aperture radius
            values.append(camera_definition[3])
            # Focus distance
            values.append(camera_definition[4])
            # Interpupillary distance
            values.append(interpupillary_distance)

        params['cameraInformation'] = values
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + 'export-frames-to-disk', params)

    def get_export_frames_progress(self):
        """
        Queries the progress of the last export of frames to disk request

        :return: Dictionary with the result: "frameNumber" with the number of
        the last written-to-disk frame, and "done", a boolean flag stating wether
        the exporting is finished or is still in progress
        :rtype: Response
        """
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + 'get-export-frames-progress')

    def cancel_frames_export(self):
        """
        Cancel the exports of frames to disk

        :return: Result of the request submission
        :rtype: Response
        """
        params = dict()
        params['path'] = '/tmp'
        params['baseName'] = ''
        params['format'] = 'png'
        params['quality'] = 100
        params['spp'] = 1
        params['startFrame'] = 0
        params['endFrame'] = 0
        params['exportIntermediateFrames'] = False
        params['animationInformation'] = []
        params['cameraInformation'] = []
        params['frameBufferMode'] = MovieMaker.FRAME_BUFFER_MODE_COLOR
        return self._client.rockets_client.request(
            self.PLUGIN_API_PREFIX + 'export-frames-to-disk', params)

    def set_current_frame(self, frame, camera_params=None):
        """
        Set the current animation frame

        :frame: Frame number
        :camera_params: Camera parameters. Defaults to None.
        """
        assert frame >= 0
        assert frame < self.get_nb_frames()

        cam = self.get_key_frame(frame)

        origin = list(cam[0])
        direction = list(cam[1])
        up = list(cam[2])

        self.set_camera(origin=origin, direction=direction, up=up)
        self._client.set_animation_parameters(current=frame)

        if camera_params is not None:
            camera_params.aperture_radius = cam[3]
            camera_params.focus_distance = cam[4]
            camera_params.enable_clipping_planes = False
            self._client.set_camera_params(camera_params)

    def display(self):
        """Displays a widget giving access to the movie frames"""
        frame = IntSlider(description='frame', min=0, max=self.get_nb_frames()-1)

        def update_frame(args):
            frame.value = args['new']
            self.set_current_frame(frame.value)

        frame.observe(update_frame, 'value')
        display(frame)

    def _set_renderer_params(self, name, samples_per_pixel, gi_length=5.0):
        """
        Set renderer with default parameters

        :name: (string): Name of the renderer
        :gi_length: (float, optional): Max length of global illumination rays. Defaults to 5.0.

        :return: Frame buffer mode (color or depth)
        :rtype: int
        """
        spp = samples_per_pixel
        frame_buffer_mode = MovieMaker.FRAME_BUFFER_MODE_COLOR
        if name == 'ambient_occlusion':
            params = self._client.AmbientOcclusionRendererParams()
            params.samples_per_frame = 16
            params.ray_length = gi_length
            self._client.set_renderer_params(params)
            spp = 4
        elif name == 'depth':
            frame_buffer_mode = MovieMaker.FRAME_BUFFER_MODE_DEPTH
            spp = 1
        elif name in ['albedo', 'raycast_Ns']:
            spp = 4
        elif name == 'shadow':
            params = self._client.ShadowRendererParams()
            params.samples_per_frame = 16
            params.ray_length = gi_length
            self._client.set_renderer_params(params)
            spp = 4
        return frame_buffer_mode, spp

    def create_snapshot(
            self, renderer, size, path, base_name, samples_per_pixel,
            export_intermediate_frames=False, gi_length=1e6, show_progress=False):
        """
        Create a snapshot of the current frame

        :renderer: Name of the renderer
        :size: Frame buffer size
        :path: Path where the snapshot file is exported
        :base_name: Base name of the snapshot file
        :samples_per_pixel: Samples per pixel
        :export_intermediate_frames: If True, intermediate samples are stored to disk. Otherwise,
        only the final accumulation is exported
        gi_length (float, optional): Max length of global illumination rays. Defaults to 5.0.
        """
        assert isinstance(size, list)
        assert isinstance(samples_per_pixel, int)
        assert len(size) == 2
        assert isinstance(export_intermediate_frames, bool)
        assert isinstance(gi_length, float)

        application_params = self._client.get_application_parameters()
        renderer_params = self._client.get_renderer()
        old_image_stream_fps = application_params['image_stream_fps']
        old_viewport_size = application_params['viewport']
        old_samples_per_pixel = renderer_params['samples_per_pixel']
        old_max_accum_frames = renderer_params['max_accum_frames']
        old_smoothed_key_frames = copy.deepcopy(self._smoothed_key_frames)

        self._client.set_renderer(current=renderer, samples_per_pixel=1, max_accum_frames=1)
        self._client.set_application_parameters(viewport=size)
        self._client.set_application_parameters(image_stream_fps=0)

        frame_buffer_mode, spp = self._set_renderer_params(renderer, samples_per_pixel, gi_length)
        self._client.set_renderer(max_accum_frames=spp)

        control_points = [self.get_camera()]
        current_animation_frame = int(self._client.get_animation_parameters()['current'])
        animation_frames = [current_animation_frame]

        self.build_camera_path(
            control_points=control_points, nb_steps_between_control_points=1, smoothing_size=1)

        if show_progress:
            progress_widget = IntProgress(description='In progress...', min=0, max=100, value=0)
            display(progress_widget)

        self.export_frames(
            path=path, base_name=base_name, animation_frames=animation_frames, size=size,
            samples_per_pixel=spp,
            export_intermediate_frames=export_intermediate_frames,
            frame_buffer_mode=frame_buffer_mode)

        done = False
        while not done:
            time.sleep(1)
            if show_progress:
                progress = self.get_export_frames_progress()['progress']
                progress_widget.value = progress * 100
            done = self.get_export_frames_progress()['done']

        if show_progress:
            progress_widget.description = 'Done'
            progress_widget.value = 100

        self._client.set_application_parameters(image_stream_fps=old_image_stream_fps,
                                                viewport=old_viewport_size)
        self._client.set_renderer(samples_per_pixel=old_samples_per_pixel,
                                  max_accum_frames=old_max_accum_frames)
        self._smoothed_key_frames = copy.deepcopy(old_smoothed_key_frames)

    def create_movie(
            self, path, size, animation_frames=list(), quality=100, samples_per_pixel=1,
            start_frame=0, end_frame=0, interpupillary_distance=0.0,
            export_intermediate_frames=True):
        """
        Create and export a set of PNG frames for later movie generation

        :path: Full path of the snapshot folder
        :size: Frame buffer size
        :animation_frames: Optional list of animation frames
        :quality: PNG quality
        :samples_per_pixel: Samples per pixel
        :start_frame: Start frame to export in the provided sequence
        :end_frame: Last frame to export in the provided sequence
        :interpupillary_distance: Interpupillary distance for stereo rendering. If set to 0, stereo
        is disabled
        :export_intermediate_frames: If True, intermediate samples are stored to disk. Otherwise,
        only the final accumulation is exported
        """
        application_params = self._client.get_application_parameters()
        renderer_params = self._client.get_renderer()

        old_image_stream_fps = application_params['image_stream_fps']
        old_viewport_size = application_params['viewport']
        old_samples_per_pixel = renderer_params['samples_per_pixel']
        old_max_accum_frames = renderer_params['max_accum_frames']
        self._client.set_renderer(samples_per_pixel=1, max_accum_frames=samples_per_pixel)
        self._client.set_application_parameters(viewport=size)
        self._client.set_application_parameters(image_stream_fps=0)

        progress_widget = IntProgress(description='In progress...', min=0, max=100, value=0)
        display(progress_widget)

        self.export_frames(
            path=path, base_name='', animation_frames=animation_frames, start_frame=start_frame,
            end_frame=end_frame, size=size, samples_per_pixel=samples_per_pixel, quality=quality,
            interpupillary_distance=interpupillary_distance,
            export_intermediate_frames=export_intermediate_frames)

        done = False
        while not done:
            time.sleep(1)
            progress = self.get_export_frames_progress()['progress']
            progress_widget.value = progress * 100
            done = self.get_export_frames_progress()['done']

        self._client.set_application_parameters(image_stream_fps=old_image_stream_fps,
                                                viewport=old_viewport_size)
        self._client.set_renderer(samples_per_pixel=old_samples_per_pixel,
                                  max_accum_frames=old_max_accum_frames)

        progress_widget.description = 'Done'
        progress_widget.value = 100
