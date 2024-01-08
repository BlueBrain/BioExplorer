/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "AnimationParameters.h"

namespace
{
constexpr auto PARAM_ANIMATION_FRAME = "animation-frame";
constexpr auto PARAM_PLAY_ANIMATION = "play-animation";
} // namespace

namespace core
{
AnimationParameters::AnimationParameters()
    : AbstractParameters("Animation")
{
    _parameters.add_options()(PARAM_ANIMATION_FRAME, po::value<uint32_t>(&_current),
                              "Scene animation frame [uint]")(PARAM_PLAY_ANIMATION,
                                                              po::bool_switch(&_playing)->default_value(false),
                                                              "Start animation playback");
}

void AnimationParameters::print()
{
    AbstractParameters::print();
    CORE_INFO("Animation frame          : " << _current);
}

void AnimationParameters::reset()
{
    _updateValue(_current, 0u, false);
    _updateValue(_dt, 0., false);
    _updateValue(_numFrames, 0u, false);
    _updateValue(_playing, false, false);
    _updateValue(_unit, std::string(), false);

    // trigger the modified callback only once
    if (isModified())
        markModified();
}

void AnimationParameters::setDelta(const int32_t delta)
{
    if (delta == 0)
        throw std::logic_error("Animation delta cannot be set to 0");
    _updateValue(_delta, delta);
}

void AnimationParameters::update()
{
    if (_playing && _canUpdateFrame())
        setFrame(getFrame() + getDelta());
}

void AnimationParameters::jumpFrames(int frames)
{
    if (_canUpdateFrame())
        setFrame(getFrame() + frames);
}

bool AnimationParameters::_canUpdateFrame() const
{
    return !hasIsReadyCallback() || _isReadyCallback();
}
} // namespace core
