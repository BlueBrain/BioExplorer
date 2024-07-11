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

#pragma once

#include "AbstractParameters.h"
#include <list>

SERIALIZATION_ACCESS(AnimationParameters)

namespace core
{
class AnimationParameters : public AbstractParameters
{
public:
    AnimationParameters();

    /** @copydoc AbstractParameters::print */
    void print() final;

    /** Reset to a 'no animation' state: 0 for dt, start and end. */
    void reset();

    /** The current frame number of the animation. */
    void setFrame(uint32_t value) { _updateValue(_current, _adjustedCurrent(value)); }
    uint32_t getFrame() const { return _current; }
    /** The (frame) delta to apply for animations to select the next frame. */
    void setDelta(const int32_t delta);
    int32_t getDelta() const { return _delta; }
    void setNumFrames(const uint32_t numFrames, const bool triggerCallback = true)
    {
        _updateValue(_numFrames, numFrames, triggerCallback);
        _updateValue(_current, std::min(_current, _numFrames), triggerCallback);
    }
    uint32_t getNumFrames() const { return _numFrames; }
    /** The dt of a simulation. */
    void setDt(const double dt, const bool triggerCallback = true) { _updateValue(_dt, dt, triggerCallback); }
    double getDt() const { return _dt; }
    /** The time unit of a simulation. */
    void setUnit(const std::string& unit, const bool triggerCallback = true)
    {
        _updateValue(_unit, unit, triggerCallback);
    }
    using IsReadyCallback = std::function<bool()>;

    /**
     * Set a callback to report if the current animation frame is ready
     * (e.g. simulation has been loaded) and the animation can advance to the
     * next frame.
     */
    void setIsReadyCallback(const IsReadyCallback& callback) { _isReadyCallback = callback; }

    /** Remove the given callback from the list of IsReadyCallbacks. */
    void removeIsReadyCallback()
    {
        if (_isReadyCallback)
        {
            reset();
            _isReadyCallback = nullptr;
        }
    }

    bool hasIsReadyCallback() const { return !!_isReadyCallback; }
    /** Update the current frame if delta is set and all listeners are ready. */
    void update();

    /** Jump 'frames' from current frame if all listeners are ready. */
    void jumpFrames(int frames);

    void togglePlayback() { _playing = !_playing; }
    bool isPlaying() const { return _playing; }

private:
    uint32_t _adjustedCurrent(const uint32_t newCurrent) const { return _numFrames == 0 ? 0 : newCurrent % _numFrames; }

    bool _canUpdateFrame() const;

    uint32_t _numFrames{0};
    uint32_t _current{0};
    int32_t _delta{1};
    bool _playing{false};
    double _dt{0};
    std::string _unit;

    IsReadyCallback _isReadyCallback;

    SERIALIZATION_FRIEND(AnimationParameters)
};
} // namespace core
