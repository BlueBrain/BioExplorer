/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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
