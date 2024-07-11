# Copyright 2020 - 2023 Blue Brain Project / EPFL
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

#                          Raphael Dumusc <raphael.dumusc@epfl.ch>
#                          Daniel Nachbaur <daniel.nachbaur@epfl.ch>
# pylint: disable=invalid-overridden-method

"""Client that connects to a remote running Brayns instance which provides the supported API."""

import asyncio
import rockets

from IPython.display import display

from .base import BaseClient
from .utils import build_schema_requests_from_registry, convert_snapshot_response_to_PIL
from . import utils


class aobject:
    """Inheriting from this class allows to define an async __init__."""

    async def __new__(cls, *a, **kw):
        """Allow to create objects by calling `await MyClass(params)`"""
        instance = super().__new__(cls)
        await instance.__init__(*a, **kw)
        return instance


class AsyncClient(BaseClient, aobject):
    """Client that connects to a remote running Brayns instance which provides the supported API."""

    async def __init__(self, url, loop=None):
        """
        Create a new client instance by connecting to the given URL.

        :param str url: a string 'hostname:port' to connect to a running Brayns instance
        :param asyncio.AbstractEventLoop loop: Event loop where this client should run in
        """
        super().__init__(url)

        self.rockets_client = rockets.AsyncClient(
            url, subprotocols=["rockets"], loop=loop
        )

        registry, requests = build_schema_requests_from_registry(self.http_url)
        schemas = await self.rockets_client.batch(requests)
        super()._build_api(registry, requests, schemas)

    # pylint: disable=W0613,W0622,E1101
    def image(
        self,
        size,
        format="jpg",
        animation_parameters=None,
        camera=None,
        quality=None,
        renderer=None,
        samples_per_pixel=None,
    ):
        """
        Request a snapshot from Brayns and return a PIL image.

        :param tuple size: (width,height) of the resulting image
        :param str format: image type as recognized by FreeImage
        :param object animation_parameters: animation params to use instead of current params
        :param object camera: camera to use instead of current camera
        :param int quality: compression quality between 1 (worst) and 100 (best)
        :param object renderer: renderer to use instead of current renderer
        :param int samples_per_pixel: samples per pixel to increase render quality
        :return: the asyncio.Future of the snapshot generation, or None if inside a notebook
        :rtype: :py:class:`~asyncio.Future`
        """
        args = locals()
        del args["self"]
        result = self.snapshot(**{k: v for k, v in args.items() if v})

        future = asyncio.get_event_loop().create_future()

        def _on_done(task):
            try:
                if task.exception():  # pragma: no cover
                    print("image() failed:", task.exception())
                else:
                    image = convert_snapshot_response_to_PIL(task.result())
                    if utils.in_notebook():  # pragma: no cover
                        if image:
                            display(image)
                    else:
                        future.set_result(image)
            except rockets.RequestError as e:  # pragma: no cover
                print("Error", e.code, e.message)
            except ConnectionRefusedError as e:  # pragma: no cover
                print(e)

        result.add_done_callback(_on_done)
        if utils.in_notebook():  # pragma: no cover
            return None
        return future
