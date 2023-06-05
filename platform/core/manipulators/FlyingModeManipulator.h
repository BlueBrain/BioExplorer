/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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


#ifndef FLYINGMODEMAINPULATOR_H
#define FLYINGMODEMAINPULATOR_H

#include "AbstractManipulator.h"

namespace core
{
/**
 * Defines a flying mode camera manipulator, like in a flight simulator.
 */
class FlyingModeManipulator : public AbstractManipulator
{
public:
    FlyingModeManipulator(Camera& camera, KeyboardHandler& keyboardHandler);
    ~FlyingModeManipulator();

private:
    void dragLeft(const Vector2i& to, const Vector2i& from) final;
    void dragRight(const Vector2i& to, const Vector2i& from) final;
    void dragMiddle(const Vector2i& to, const Vector2i& from) final;
    void wheel(const Vector2i& position, float delta) final;

    void _strafeLeft();
    void _strafeRight();
    void _flyForward();
    void _flyBackwards();
};
} // namespace core

#endif
