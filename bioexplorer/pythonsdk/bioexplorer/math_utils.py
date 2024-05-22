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

import math
from pyquaternion import Quaternion


class Vector3:
    """A 3D vector class for performing vector operations."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
        Initialize a Vector3 instance with x, y, and z components.

        :param x: x-component of the vector
        :param y: y-component of the vector
        :param z: z-component of the vector
        """
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        """Returns a string representation of the vector."""
        return f"[{self.x}, {self.y}, {self.z}]"

    def __repr__(self):
        """Return the official string representation of the vector."""
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __add__(self, other):
        """Add two vectors."""
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Subtract two vectors."""
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        """Multiply vector by a scalar or dot-multiply with another vector."""
        if isinstance(other, (int, float)):  # Scalar multiplication
            return Vector3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3):  # Dot product
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Vector3' and '{}'".format(type(other)))

    def __rmul__(self, other):
        """Right multiplication to support scalar multiplication with scalar on the left."""
        return self.__mul__(other)

    def cross(self, other):
        """Cross product of two vectors."""
        return Vector3(self.y * other.z - self.z * other.y,
                       self.z * other.x - self.x * other.z,
                       self.x * other.y - self.y * other.x)

    def magnitude(self):
        """Return the magnitude (length) of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        """Normalize the vector."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return self * (1.0 / mag)

    def to_list(self):
        """Convert vector to a list."""
        return [self.x, self.y, self.z]

    def copy(self):
        """Return a copy of the vector."""
        return Vector3(self.x, self.y, self.z)


class Vector2:
    """A 2D vector class for performing vector operations."""

    def __init__(self, x=0.0, y=0.0):
        """
        Initialize a Vector2 instance with x and y components.

        :param x: x-component of the vector
        :param y: y-component of the vector
        """
        self.x = x
        self.y = y

    def __str__(self):
        """Returns a string representation of the vector."""
        return f"[{self.x}, {self.y}]"

    def __repr__(self):
        """Return the official string representation of the vector."""
        return f"Vector2({self.x}, {self.y})"

    def __add__(self, other):
        """Add two vectors."""
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Subtract two vectors."""
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        """Multiply vector by a scalar or dot-multiply with another vector."""
        if isinstance(other, (int, float)):  # Scalar multiplication
            return Vector2(self.x * other, self.y * other)
        elif isinstance(other, Vector2):  # Dot product
            return self.x * other.x + self.y * other.y
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector2' and '{type(other).__name__}'")

    def __rmul__(self, other):
        """Right multiplication to support scalar multiplication with scalar on the left."""
        return self.__mul__(other)

    def perpendicular(self):
        """Return a vector that is perpendicular to this one."""
        return Vector2(-self.y, self.x)

    def magnitude(self):
        """Return the magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Normalize the vector."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return self * (1.0 / mag)

    def to_list(self):
        """Convert vector to a list."""
        return [self.x, self.y]

    def copy(self):
        """Return a copy of the vector."""
        return Vector2(self.x, self.y)


class Bounds:
    """Bounds of a 3D object, represented by an axis-aligned bounding box."""

    def __init__(self, min_aabb, max_aabb):
        """
        Initialize the bounding box with minimum and maximum coordinates.

        :param min_aabb: Vector3 representing the minimum coordinates.
        :param max_aabb: Vector3 representing the maximum coordinates.
        """
        assert isinstance(min_aabb, Vector3), "min_aabb must be an instance of Vector3"
        assert isinstance(max_aabb, Vector3), "max_aabb must be an instance of Vector3"
        self.min_aabb = min_aabb
        self.max_aabb = max_aabb
        self.center = Vector3()
        self.size = Vector3()
        self.update()

    def update(self):
        """Recalculate the center and size of the bounding box based on min/max coordinates."""
        self.center = Vector3(
            (self.min_aabb.x + self.max_aabb.x) / 2,
            (self.min_aabb.y + self.max_aabb.y) / 2,
            (self.min_aabb.z + self.max_aabb.z) / 2
        )
        self.size = Vector3(
            self.max_aabb.x - self.min_aabb.x,
            self.max_aabb.y - self.min_aabb.y,
            self.max_aabb.z - self.min_aabb.z
        )

    def __str__(self):
        """Returns a stringified representation of the bounding box."""
        return f"Bounds(min={self.min_aabb}, max={self.max_aabb}, center={self.center}, size={self.size})"

    def contains(self, point):
        """
        Check if a point is within the bounding box.

        :param point: Vector3 point to check.
        :return: True if point is within the bounds, else False.
        """
        return (self.min_aabb.x <= point.x <= self.max_aabb.x and
                self.min_aabb.y <= point.y <= self.max_aabb.y and
                self.min_aabb.z <= point.z <= self.max_aabb.z)

    def intersects(self, other):
        """
        Check if this bounding box intersects with another bounding box.

        :param other: Another Bounds object to check against.
        :return: True if the bounding boxes intersect, else False.
        """
        return (self.min_aabb.x <= other.max_aabb.x and self.max_aabb.x >= other.min_aabb.x and
                self.min_aabb.y <= other.max_aabb.y and self.max_aabb.y >= other.min_aabb.y and
                self.min_aabb.z <= other.max_aabb.z and self.max_aabb.z >= other.min_aabb.z)

    def copy(self):
        """Return a copy of the bounding box."""
        return Bounds(self.min_aabb.copy(), self.max_aabb.copy())


class Transformation:
    """Transformation defined by translation, rotation, rotation center, and scale."""

    def __init__(self, translation=Vector3(), rotation=Quaternion(), rotation_center=Vector3(), scale=Vector3(1, 1, 1)):
        assert isinstance(translation, Vector3)
        assert isinstance(rotation, Quaternion)
        assert isinstance(rotation_center, Vector3)
        assert isinstance(scale, Vector3)
        self.translation = translation
        self.rotation = rotation
        self.rotation_center = rotation_center
        self.scale = scale

    def apply(self, point):
        """Apply the transformation to a point."""
        # Convert point to Vector3 if not already (optional, depends on use case)
        if not isinstance(point, Vector3):
            point = Vector3(point[0], point[1], point[2])

        # Move the point to the rotation center
        point = point - self.rotation_center
        # Apply rotation
        rotated_point = self.rotation.rotate(point.to_list())
        point = Vector3(rotated_point[0], rotated_point[1], rotated_point[2])
        # Scale the point
        point = Vector3(point.x * self.scale.x, point.y * self.scale.y, point.z * self.scale.z)
        # Move back from rotation center and apply translation
        point = point + self.rotation_center + self.translation
        return point

    def __repr__(self):
        return f"Transformation(translation={self.translation}, rotation={self.rotation}, rotation_center={self.rotation_center}, scale={self.scale})"
