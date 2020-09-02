#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of BioExplorer
# <https://github.com/BlueBrain/BioExplorer>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# All rights reserved. Do not distribute without further notice.

"""Provides a class that ease the definition of smoothed camera paths"""

from .bio_explorer import BioExplorer


class MovieMaker:
    """ Movie maker """

    def __init__(self, bioexplorer):
        assert isinstance(bioexplorer, BioExplorer)
        self._be = bioexplorer
        self._client = bioexplorer.core_api()
        self._smoothed_key_frames = list()

    def build_camera_path(self, control_points, nb_steps_between_control_points, smoothing_size=1):
        """
        Build a camera path from control points

        @param control_points: List of control points
        @param nb_steps_between_control_points: Number of steps between two control points
        @param smoothing_size: Number of steps to be considered for the smoothing of the path
        """
        origins = list()
        directions = list()
        ups = list()
        aperture_radii = list()
        focus_distances = list()

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
                    t_origin[k] = (p1['origin'][k] - p0['origin'][k]) / float(nb_steps_between_control_points)
                    t_direction[k] = (p1['direction'][k] - p0['direction'][k]) / float(nb_steps_between_control_points)
                    t_up[k] = (p1['up'][k] - p0['up'][k]) / float(nb_steps_between_control_points)

                    origin[k] = p0['origin'][k] + t_origin[k] * float(i)
                    direction[k] = p0['direction'][k] + t_direction[k] * float(i)
                    up[k] = p0['up'][k] + t_up[k] * float(i)

                t_aperture_radius = (p1['apertureRadius'] - p0['apertureRadius']) / float(
                    nb_steps_between_control_points)
                aperture_radius = p0['apertureRadius'] + t_aperture_radius * float(i)

                t_focus_distance = (p1['focusDistance'] - p0['focusDistance']) / float(nb_steps_between_control_points)
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
            self._smoothed_key_frames.append([(o[0] / smoothing_size, o[1] / smoothing_size, o[2] / smoothing_size),
                                              (d[0] / smoothing_size, d[1] / smoothing_size, d[2] / smoothing_size),
                                              (u[0] / smoothing_size, u[1] / smoothing_size, u[2] / smoothing_size),
                                              aperture_radius / smoothing_size, focus_distance / smoothing_size])
        last = control_points[len(control_points) - 1]
        self._smoothed_key_frames.append(
            (last['origin'], last['direction'], last['up'], last['apertureRadius'], last['focusDistance']))

    def get_nb_frames(self):
        """
        Get the number of smoothed frames

        @return: The number of smoothed frames
        """
        return len(self._smoothed_key_frames)

    def get_key_frame(self, frame):
        """
        Get the smoothed camera information for the given frame

        @param int frame: Frame number
        @return: The smoothed camera information for the given frame
        """
        if frame < len(self._smoothed_key_frames):
            return self._smoothed_key_frames[frame]
        raise KeyError

    def set_camera(self, origin, direction, up):
        """
        Set the camera using origin, direction and up vectors

        @param list origin: Origin of the camera
        @param list direction: Direction in which the camera is looking
        @param list up: Up vector
        @return: Result of the request submission
        """
        params = dict()
        params['origin'] = origin
        params['direction'] = direction
        params['up'] = up
        return self._client.rockets_client.request('set-odu-camera', params)

    def get_camera(self):
        """
        Get the origin, direction and up vector of the camera

        @return: A JSon representation of the origin, direction and up vectors
        """
        return self._client.rockets_client.request('get-odu-camera')

    def export_frames(self, path, size, image_format='png',
                      quality=100, samples_per_pixel=1, start_frame=0):
        """
        Exports frames to disk. Frames are named using a 6 digit representation of the frame number

        @param path: Folder into which frames are exported
        @param image_format: Image format (the ones supported par Brayns: PNG, JPEG, etc)
        @param quality: Quality of the exported image (Between 0 and 100)
        @param samples_per_pixel: Number of samples per pixels
        @param start_frame: Optional value if the rendering should start at a specific frame.
        This is used to resume the rendering of a previously canceled sequence)
        @return: Result of the request submission
        """

        assert isinstance(size, list)
        assert len(size) == 2

        self._client.set_application_parameters(viewport=size)
        self._client.set_renderer(accumulation=True, samples_per_pixel=1, max_accum_frames=samples_per_pixel + 1,
                                  subsampling=1)

        animation_frames = list()
        camera_definitions = list()
        for i in range(self.get_nb_frames()):
            animation_frames.append(0)
            camera_definitions.append(self.get_key_frame(i))

        params = dict()
        params['path'] = path
        params['format'] = image_format
        params['quality'] = quality
        params['spp'] = samples_per_pixel
        params['startFrame'] = start_frame
        params['animationInformation'] = animation_frames
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
        params['cameraInformation'] = values
        self._client.rockets_client.request('export-frames-to-disk', params)

        # Wait until all frames are rendered
        while self.get_export_frames_progress()['progress'] < 1.0:
            import time
            time.sleep(1)

        # Set back-end not to export frames anymore
        self.cancel_frames_export()

    def get_export_frames_progress(self):
        """
        Queries the progress of the last export of frames to disk request

        @return: Dictionary with the result: "frameNumber" with the number of
        the last written-to-disk frame, and "done", a boolean flag stating wether
        the exporting is finished or is still in progress
        """
        return self._client.rockets_client.request('get-export-frames-progress')

    def cancel_frames_export(self):
        """
        Cancel the exports of frames to disk

        @return: Result of the request submission
        """
        params = dict()
        params['path'] = '/tmp'
        params['format'] = 'png'
        params['quality'] = 100
        params['spp'] = 1
        params['startFrame'] = 0
        params['animationInformation'] = []
        params['cameraInformation'] = []
        return self._client.rockets_client.request('export-frames-to-disk', params)
