/*
 * Copyright (c) 2015-2017, EPFL/Blue Brain Project
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


#include "BaseWindow.h"

#include <platform/core/Core.h>
#include <platform/core/common/Logs.h>
#include <platform/core/common/input/KeyboardHandler.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Renderer.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/manipulators/AbstractManipulator.h>
#include <platform/core/parameters/ParametersManager.h>

#include <assert.h>

#ifdef __APPLE__
#include "GLUT/glut.h"
#include <unistd.h>
#else
#include "GL/glut.h"
#include <GL/freeglut_ext.h>
#endif

namespace
{
const int GLUT_WHEEL_SCROLL_UP = 3;
const int GLUT_WHEEL_SCROLL_DOWN = 4;
} // namespace

namespace core
{
uint64_t currentFrame = 0;
uint64_t maxFrame = 0;

void runGLUT()
{
    glutMainLoop();
}

void initGLUT(int* ac, const char** av)
{
    glutInit(ac, (char**)av);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
}

// ------------------------------------------------------------------
// glut event handlers
// ------------------------------------------------------------------
void glut3dReshape(int x, int y)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->reshape(Vector2i(x, y));
}

void glut3dDisplay(void)
{
    if (BaseWindow::_activeWindow)
    {
        BaseWindow::_activeWindow->display();
    }
}

void glut3dKeyboard(unsigned char key, int x, int y)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->keypress(key, Vector2i(x, y));
}
void glut3dSpecial(int key, int x, int y)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->specialkey(key, Vector2i(x, y));
}

void glut3dIdle(void)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->idle();
}
void glut3dMotionFunc(int x, int y)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->motion(Vector2i(x, y));
}

void glut3dMouseFunc(int whichButton, int released, int x, int y)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->mouseButton(whichButton, released, Vector2i(x, y));
}

void glut3dPassiveMouseFunc(int x, int y)
{
    if (BaseWindow::_activeWindow)
        BaseWindow::_activeWindow->passiveMotion(Vector2i(x, y));
}

// ------------------------------------------------------------------
// Base window
// ------------------------------------------------------------------
/*! currently active window */
BaseWindow* BaseWindow::_activeWindow = nullptr;

BaseWindow::BaseWindow(Core& core, const FrameBufferMode frameBufferMode)
    : _core(core)
    , _lastMousePos(-1, -1)
    , _currMousePos(-1, -1)
    , _lastButtonState(0)
    , _currButtonState(0)
    , _currModifiers(0)
    , _frameBufferMode(frameBufferMode)
    , _windowID(-1)
    , _windowSize(-1, -1)
    , _displayHelp(false)
    , _fullScreen(false)
{
    const auto motionSpeed = _core.getCameraManipulator().getMotionSpeed();
    CORE_INFO("Camera       :" << _core.getEngine().getCamera());
    CORE_INFO("Motion speed :" << motionSpeed);

    const auto& rp = _core.getParametersManager().getRenderingParameters();
    if (rp.getAccumulationType() == AccumulationType::ai_denoised)
        _frameBufferMode = FrameBufferMode::COLOR_F32;

    _rendererTypes = _core.getEngine().getRendererTypes();
}

BaseWindow::~BaseWindow() {}

void BaseWindow::mouseButton(const int button, const bool released, const Vector2i& pos)
{
    if (pos != _currMousePos)
        motion(pos);
    _lastButtonState = _currButtonState;

    if (released)
        _currButtonState = _currButtonState & ~(1 << button);
    else
        _currButtonState = _currButtonState | (1 << button);
    _currModifiers = glutGetModifiers();

    auto& manipulator = _core.getCameraManipulator();

    if (_currModifiers & GLUT_ACTIVE_SHIFT && released)
    {
        const auto& result =
            _core.getEngine().getRenderer().pick({pos.x / float(_windowSize.x), 1.f - pos.y / float(_windowSize.y)});
        _core.getEngine().getFrameBuffer().clear();
        if (result.hit)
        {
            _core.getEngine().getCamera().setTarget(result.pos);
            // updates position based on new target and current rotation
            manipulator.rotate(result.pos, 0, 0, AbstractManipulator::AxisMode::localY);
        }
    }

    if (button == GLUT_WHEEL_SCROLL_UP || button == GLUT_WHEEL_SCROLL_DOWN)
    {
        // Wheel events are reported twice like a button click (press + release)
        if (released)
            return;
        const auto delta = (button == GLUT_WHEEL_SCROLL_UP) ? 1 : -1;
        manipulator.wheel(pos, delta);
    }
}

void BaseWindow::motion(const Vector2i& pos)
{
    _currMousePos = pos;
    if (_currButtonState != _lastButtonState)
    {
        // some button got pressed; reset 'old' pos to new pos.
        _lastMousePos = _currMousePos;
        _lastButtonState = _currButtonState;
    }

    auto& manipulator = _core.getCameraManipulator();

    if ((_currButtonState == (1 << GLUT_RIGHT_BUTTON)) ||
        ((_currButtonState == (1 << GLUT_LEFT_BUTTON)) && (_currModifiers & GLUT_ACTIVE_ALT)))
    {
        manipulator.dragRight(_currMousePos, _lastMousePos);
    }
    else if ((_currButtonState == (1 << GLUT_MIDDLE_BUTTON)) ||
             ((_currButtonState == (1 << GLUT_LEFT_BUTTON)) && (_currModifiers & GLUT_ACTIVE_CTRL)))
    {
        manipulator.dragMiddle(_currMousePos, _lastMousePos);
    }
    else if (_currButtonState == (1 << GLUT_LEFT_BUTTON))
    {
        manipulator.dragLeft(_currMousePos, _lastMousePos);
    }

    _lastMousePos = _currMousePos;
}

void BaseWindow::passiveMotion(const Vector2i& pos)
{
    _mouse = pos;
}

void BaseWindow::idle()
{
    usleep(1000);
}

void BaseWindow::reshape(const Vector2i& newSize)
{
    _windowSize = newSize;

    auto& applicationParameters = _core.getParametersManager();
    applicationParameters.getApplicationParameters().setWindowSize(_windowSize);
}

void BaseWindow::activate()
{
    _activeWindow = this;
    glutSetWindow(_windowID);
}

void BaseWindow::forceRedraw()
{
    glutPostRedisplay();
}

void BaseWindow::display()
{
    const Vector2ui windowSize = _core.getParametersManager().getApplicationParameters().getWindowSize();
    if (windowSize != _windowSize)
        glutReshapeWindow(windowSize.x, windowSize.y);

    _timer.start();

    const auto& camera = _core.getEngine().getCamera();
    _renderInput.windowSize = windowSize;
    _renderInput.position = camera.getPosition();
    _renderInput.orientation = camera.getOrientation();
    _renderInput.target = camera.getTarget();

    const auto& fb = _core.getEngine().getFrameBuffer();
    const auto& rp = _core.getParametersManager().getRenderingParameters();
    const auto maxAccumFrames = rp.getMaxAccumFrames();

    if (fb.numAccumFrames() < maxAccumFrames)
        _core.commitAndRender(_renderInput, _renderOutput);
    else
        _core.commit();

    GLenum format = GL_RGBA;
    switch (_renderOutput.colorBufferFormat)
    {
    case FrameBufferFormat::bgra_i8:
        format = GL_BGRA;
        break;
    case FrameBufferFormat::rgb_i8:
        format = GL_RGB;
        break;
    default:
        format = GL_RGBA;
    }

    GLenum type = GL_FLOAT;
    GLvoid* buffer = nullptr;
    switch (_frameBufferMode)
    {
    case FrameBufferMode::COLOR_I8:
        type = GL_UNSIGNED_BYTE;
        buffer = _renderOutput.colorBuffer.data();
        break;
    case FrameBufferMode::COLOR_F32:
    {
        type = GL_FLOAT;
        // format = GL_BGRA;
        format = GL_RGBA;
        buffer = _renderOutput.floatBuffer.data();
        break;
    }
    case FrameBufferMode::DEPTH_F32:
        format = GL_LUMINANCE;
        buffer = _renderOutput.floatBuffer.data();
        break;
    default:
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    if (buffer)
        glDrawPixels(_windowSize.x, _windowSize.y, format, type, buffer);

    const auto time = std::chrono::steady_clock::now();
    const uint64_t millisecondsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time - _chrono).count();
    if (millisecondsElapsed < _hintDelay)
        _renderBitmapString(0.f, -0.9f, _hintMessage);
    else
        _hintMessage = "";

    if (_displayHelp)
    {
        auto& keyHandler = _core.getKeyboardHandler();
        std::string help;
        for (const auto& value : keyHandler.help())
            help += value + "\n";
        _renderBitmapString(0.f, 0.8f, help);
    }

    _timer.stop();

    glutSwapBuffers();

    clearPixels();

    forceRedraw();
}

void BaseWindow::clearPixels()
{
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void BaseWindow::drawPixels(const int* framebuffer)
{
    glDrawPixels(_windowSize.x, _windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer);
    glutSwapBuffers();
}

void BaseWindow::drawPixels(const Vector3f* framebuffer)
{
    glDrawPixels(_windowSize.x, _windowSize.y, GL_RGBA, GL_FLOAT, framebuffer);
    glutSwapBuffers();
}

void BaseWindow::_setTitle(const char* title)
{
    assert(_windowID >= 0);
    glutSetWindow(_windowID);
    glutSetWindowTitle(title);
}

void BaseWindow::create(const char* title, const size_t width, const size_t height)
{
    glutInitWindowSize(width, height);
    _windowID = glutCreateWindow(title);
    _activeWindow = this;
    glutDisplayFunc(glut3dDisplay);
    glutReshapeFunc(glut3dReshape);
    glutKeyboardFunc(glut3dKeyboard);
    glutSpecialFunc(glut3dSpecial);
    glutMotionFunc(glut3dMotionFunc);
    glutMouseFunc(glut3dMouseFunc);
    glutPassiveMotionFunc(glut3dPassiveMouseFunc);
    glutIdleFunc(glut3dIdle);

    _registerKeyboardShortcuts();
}

void BaseWindow::keypress(const char key, const Vector2f&)
{
    switch (key)
    {
    case 'h':
        _displayHelp = !_displayHelp;
        break;
    case '$':
        _setHint("My mind won't fit on a server somewhere I could never afford");
        break;
    case '*':
        _setHint("You can't download the sun. You'll never download me");
        break;
    case 27:
    case 'Q':
#ifdef __APPLE__
        exit(0);
#else
        glutLeaveMainLoop();
#endif
        break;
    default:
        auto& kh = _core.getKeyboardHandler();
        kh.handleKeyboardShortcut(key);
        if (_hintMessage.empty())
            _setHint(kh.getKeyboardShortcutDescription(key));
    }

    _core.getEngine().commit();
}

void BaseWindow::specialkey(const int key, const Vector2f&)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:
        _core.getKeyboardHandler().handle(SpecialKey::LEFT);
        break;
    case GLUT_KEY_RIGHT:
        _core.getKeyboardHandler().handle(SpecialKey::RIGHT);
        break;
    case GLUT_KEY_UP:
        _core.getKeyboardHandler().handle(SpecialKey::UP);
        break;
    case GLUT_KEY_DOWN:
        _core.getKeyboardHandler().handle(SpecialKey::DOWN);
        break;
    case GLUT_KEY_F11:
        if (_fullScreen)
            glutPositionWindow(_windowPosition.x, _windowPosition.y);
        else
        {
            _windowPosition.x = glutGet((GLenum)GLUT_WINDOW_X);
            _windowPosition.y = glutGet((GLenum)GLUT_WINDOW_Y);
            glutFullScreen();
        }
        _fullScreen = !_fullScreen;
        break;
    }
}

void BaseWindow::_registerKeyboardShortcuts()
{
    auto& keyHandler = _core.getKeyboardHandler();
    keyHandler.registerKeyboardShortcut('z', "Switch between depth and color buffers",
                                        std::bind(&BaseWindow::_toggleFrameBuffer, this));
    keyHandler.registerKeyboardShortcut('n', "Next renderer type", std::bind(&BaseWindow::_toggleRendererType, this));
    keyHandler.registerKeyboardShortcut('l', "Toggle head light", std::bind(&BaseWindow::_toggleHeadLight, this));
}

#ifdef __APPLE__
void BaseWindow::_renderBitmapString(const float, const float, const std::string&) {}
#else
void BaseWindow::_renderBitmapString(const float x, const float y, const std::string& text)
{
    const unsigned char* msg = reinterpret_cast<const unsigned char*>(text.c_str());
    const auto font = GLUT_BITMAP_HELVETICA_18;
    const float normalizeTextLength =
        static_cast<float>(glutBitmapLength(font, msg)) / static_cast<float>(_windowSize.x);
    const float normalizeTextHeight = static_cast<float>(glutBitmapHeight(font)) / static_cast<float>(_windowSize.y);
    glRasterPos2f(x - normalizeTextLength, y - normalizeTextHeight);
    glutBitmapString(font, msg);
    glRasterPos2f(-1.f, -1.f);
}
#endif

void BaseWindow::_toggleFrameBuffer()
{
    size_t mode = static_cast<size_t>(_frameBufferMode);
    mode = (mode + 1) % 2;
    auto& engine = _core.getEngine();
    const auto& ap = engine.getParametersManager().getApplicationParameters();
    const auto& engineName = ap.getEngine();
    if (engineName == ENGINE_OSPRAY && mode == static_cast<size_t>(AccumulationType::ai_denoised))
        mode = (mode + 1) % 2;
    _frameBufferMode = static_cast<FrameBufferMode>(mode);

    // Accumulation type
    auto& frameBuffer = engine.getFrameBuffer();
    if (_frameBufferMode == FrameBufferMode::COLOR_F32)
    {
        const auto& rp = engine.getParametersManager().getRenderingParameters();
        if (rp.getToneMapperExposure() > 0.f)
            _setHint("Post processing: AI Denoiser + Tone mapper");
        else
            _setHint("Post processing: AI Denoiser");
        frameBuffer.setAccumulationType(AccumulationType::ai_denoised);
    }
    else
    {
        _setHint("Post processing: None");
        frameBuffer.setAccumulationType(AccumulationType::linear);
    }
}

void BaseWindow::_toggleRendererType()
{
    ++_currentRendererTypeIndex;
    _currentRendererTypeIndex = _currentRendererTypeIndex % _rendererTypes.size();
    const auto rendererType = _rendererTypes[_currentRendererTypeIndex];
    auto& rp = _core.getParametersManager().getRenderingParameters();
    rp.setCurrentRenderer(rendererType);
    _setHint("Renderer: [" + rendererType + "]");
}

void BaseWindow::_toggleHeadLight()
{
    auto& rp = _core.getParametersManager().getRenderingParameters();
    rp.setHeadLight(!rp.getHeadLight());
    std::string hint = "Head light: [";
    hint += (rp.getHeadLight() ? "ON" : "OFF");
    hint += "]";
    _setHint(hint);
}

void BaseWindow::_setHint(const std::string& message, const uint64_t milliseconds)
{
    _hintMessage = message;
    _hintDelay = milliseconds;
    _chrono = std::chrono::steady_clock::now();
}

} // namespace core
