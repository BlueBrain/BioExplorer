/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "AbstractManipulator.h"

namespace core
{
/**
 * Camera manipulator to rotate and zoom around a central point.
 */
class InspectCenterManipulator : public AbstractManipulator
{
public:
    InspectCenterManipulator(Camera& camera, KeyboardHandler& handler);
    ~InspectCenterManipulator();

private:
    void dragLeft(const Vector2i& to, const Vector2i& from) final;
    void dragRight(const Vector2i& to, const Vector2i& from) final;
    void dragMiddle(const Vector2i& to, const Vector2i& from) final;
    void wheel(const Vector2i& position, float delta) final;

    void _rotateLeft();
    void _rotateRight();
    void _rotateUp();
    void _rotateDown();

    void _turnLeft();
    void _turnRight();
    void _turnUp();
    void _turnDown();
};
} // namespace core
