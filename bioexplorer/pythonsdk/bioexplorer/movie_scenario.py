#!/usr/bin/env python
"""Movie scenario"""

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

from datetime import datetime, timedelta
import time
import math
import argparse
import os
from .bio_explorer import BioExplorer
from .movie_maker import MovieMaker

# pylint: disable=no-member
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


class MovieScenario:
    """Super class for creating movies"""

    def __init__(self, hostname, port, projection, output_folder, image_k=4,
                 image_samples_per_pixel=64, log_level=1, v1_compatibility=False,
                 shaders=list(['bio_explorer']), draft=False, gi_length=1e6):
        """
        Initialize movie scenario

        :hostname: Host name where BioExplorer is running
        :port: Port of the BioExplorer WebSocket
        :projection: Camera projection (Perspective, Orthographic, etc)
        :output_folder: Folder where frame are savec
        :image_k: Number of 'k' defining the frame resolution. Defaults to 4.
        :image_samples_per_pixel: Number of samples per pixel. Defaults to 64.
        :log_level: Logging level. Defaults to 1
        :v1_compatibility: Used to regenerated movies from BioExplorer version 1. Defaults to False.
        :shaders: List of shaders to render. Defaults to list(['bio_explorer']).
        """
        self._log_level = log_level
        self._hostname = hostname
        self._url = hostname + ':' + str(port)
        self._be = BioExplorer(self._url)
        self._core = self._be.core_api()
        self._mm = MovieMaker(self._be)

        self._image_size = [image_k * 960, image_k * 540]
        self._image_samples_per_pixel = image_samples_per_pixel
        self._image_projection = projection
        self._image_output_folder = output_folder
        self._shaders = shaders
        self._gi_length = gi_length
        self._draft = draft
        self._prepare_movie(projection, image_k)
        self._be.set_general_settings(
            model_visibility_on_creation=False,
            v1_compatibility=v1_compatibility
        )
        self._log(1, '============================================================================')
        self._log(1, '- Version          : ' + self._be.version())
        self._log(1, '- URL              : ' + self._url)
        self._log(1, '- Projection       : ' + projection)
        self._log(1, '- Frame size       : ' + str(self._image_size))
        self._log(1, '- Export folder    : ' + self._image_output_folder)
        self._log(1, '- Samples per pixel: ' + str(self._image_samples_per_pixel))
        self._log(1, '============================================================================')

    def build_frame(self, frame):
        """
        Build the specified frame

        :frame: Index of the frame to build
        """
        raise NotImplementedError('You need to define a build_frame method!')

    def render_movie(
            self, cameras_key_frames, nb_frames_between_keys, nb_smoothing_frames, start_frame=0,
            end_frame=0, frame_step=1, frame_list=list()):
        """
        Render the speficied frames

        :cameras_key_frames: List of camera key-frames
        :nb_frames_between_keys: Number of frames between each key frame
        :nb_smoothing_frames: Number of frames used to average the camera position and orientation
        :start_frame: Index of the first frame to render. Defaults to 0
        :end_frame: Index of the last frame to render. Defaults to 0
        :frame_step: Number of frames to skip. Defaults to 1
        :frame_list: Explicit list of frames to render. Defaults to list().
        """
        self._mm.build_camera_path(cameras_key_frames, nb_frames_between_keys, nb_smoothing_frames)
        self._log(1, '- Total number of frames: %d' % self._mm.get_nb_frames())

        self._core.set_application_parameters(viewport=self._image_size)
        self._core.set_application_parameters(image_stream_fps=0)

        frames_to_render = list()
        if len(frame_list) != 0:
            frames_to_render = frame_list
        else:
            if end_frame == 0:
                end_frame = self._mm.get_nb_frames()
            for i in range(start_frame, end_frame, frame_step):
                frames_to_render.append(i)

        cumulated_rendering_time = 0
        nb_frames = len(frames_to_render)
        frame_count = 1

        # Frames
        for frame in frames_to_render:
            start = time.time()
            self._log(1, '- Rendering frame %i (%i/%i)' % (frame, frame_count, nb_frames))
            self._log(1, '------------------------------')

            # Stop rendering during the loading of the scene
            self._core.set_renderer(
                samples_per_pixel=1, subsampling=1, max_accum_frames=1)

            # Set camera
            self._mm.set_current_frame(
                frame=frame, camera_params=self._core.BioExplorerPerspectiveCameraParams())

            # Frame setup
            self.build_frame(frame)

            self._log(1, '- Frame buffers')
            for shader in self._shaders:
                # Rendering settings
                self._log(2, '-   ' + shader)
                self._render_frame(shader, frame)

            end = time.time()

            rendering_time = end - start
            cumulated_rendering_time += rendering_time
            average_rendering_time = cumulated_rendering_time / frame_count
            remaining_rendering_time = (nb_frames - frame_count) * average_rendering_time
            self._log(1, '------------------------------')
            self._log(1, 'Frame %i successfully rendered in %i seconds' %
                      (frame, rendering_time))

            hours = math.floor(remaining_rendering_time / 3600)
            minutes = math.floor((remaining_rendering_time - hours * 3600) / 60)
            seconds = math.floor(remaining_rendering_time - hours * 3600 - minutes * 60)

            expected_end_time = datetime.now() + timedelta(seconds=remaining_rendering_time)
            self._log(1, 'Estimated remaining time: %i hours, %i minutes, %i seconds' %
                      (hours, minutes, seconds))
            self._log(1, 'Expected end time       : %s' % expected_end_time)
            self._log(
                1, '----------------------------------------------------------------------------')
            frame_count += 1

        self._core.set_application_parameters(image_stream_fps=20)
        self._log(1, 'Movie rendered, live long and prosper \\V/')

    def _log(self, level, message):
        if level <= self._log_level:
            print('[' + str(datetime.now()) + '] ' + message)

    @staticmethod
    def _check(method):
        response = method
        if not response['status']:
            raise Exception(response['contents'])

    def _make_export_folders(self):
        for folder in self._shaders:
            path = self._image_output_folder + '/' + folder
            if not os.path.isdir(path):
                self._log(1, 'Creating ' + path)
                os.makedirs(path)

    def _prepare_movie(self, projection, image_k):
        if projection == 'perspective':
            self._image_size = [image_k*960, image_k*540]
            self._core.set_camera(current='bio_explorer_perspective')
        elif projection == 'fisheye':
            self._image_size = [int(image_k*1024), int(image_k*1024)]
            self._core.set_camera(current='fisheye')
        elif projection == 'panoramic':
            self._image_size = [int(image_k*1024), int(image_k*1024)]
            self._core.set_camera(current='panoramic')
        elif projection == 'opendeck':
            self._log(1, 'Warning: OpenDeck resolution is set server side '
                         '(--resolution-scaling plug-in parameter)')
            self._image_size = [11940, 3424]
            self._core.set_camera(current='cylindric')

        self._image_output_folder = self._image_output_folder + '/' + \
            projection + '/' + str(self._image_size[0]) + 'x' + str(self._image_size[1])
        self._make_export_folders()
        self._log(2, '- Forcing viewport size to ' + str(self._image_size))
        self._core.set_application_parameters(viewport=self._image_size)

    def _render_frame(self, renderer, frame):
        self._log(2, '- Creating snapshot ' + str(self._image_size))
        self._mm.create_snapshot(
            renderer=renderer,
            size=self._image_size, path=self._image_output_folder + '/' + renderer,
            base_name='%05d' % frame, samples_per_pixel=self._image_samples_per_pixel,
            gi_length=self._gi_length)

    @staticmethod
    def parse_arguments(argv):
        """
        Parse command line arguments

        :argv: List of command line arguments
        """
        parser = argparse.ArgumentParser(description='Missing frames')
        parser.add_argument('-e', '--export-folder', help='Export folder', type=str, default='/tmp')
        parser.add_argument('-n', '--hostname',
                            help='BioExplorer server hostname', type=str, default='localhost')
        parser.add_argument('-p', '--port',
                            help='BioExplorer server port', type=int, default=5000)
        parser.add_argument('-j', '--projection', help='Camera projection',
                            type=str, default='perspective',
                            choices=['perspective', 'fisheye', 'panoramic', 'opendeck'])
        parser.add_argument('-r', '--shaders', help='Camera projection',
                            type=str, nargs='*', default=['bio_explorer'],
                            choices=[
                                'albedo', 'ambient_occlusion', 'basic', 'depth', 'raycast_Ns',
                                'bio_explorer', 'circuit_explorer_advanced'])
        parser.add_argument('-k', '--image-resolution-k',
                            help='Image resolution in K', type=int, default=1)
        parser.add_argument('-s', '--image-samples-per-pixel',
                            help='Image samples per pixel', type=int, default=32)
        parser.add_argument('-f', '--from_frame', type=int, help='Start frame', default=0)
        parser.add_argument('-t', '--to-frame', type=int, help='End frame', default=0)
        parser.add_argument('-m', '--frame-step', type=int, help='Frame step', default=1)
        parser.add_argument('-g', '--log-level', type=int, help='Frame step', default=1)
        parser.add_argument('-l', '--frame-list', type=int, nargs='*',
                            help='List of frames to render', default=list())
        parser.add_argument('-z', '--magnetic', help='Magnetic fields', action='store_true')
        parser.add_argument('-d', '--draft', help='Draft mode', action='store_true')
        return parser.parse_args(argv)
