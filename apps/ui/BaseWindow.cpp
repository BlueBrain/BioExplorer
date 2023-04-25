/* Copyright (c) 2015-2017, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "BaseWindow.h"

#include <brayns/Brayns.h>
#include <brayns/common/input/KeyboardHandler.h>
#include <brayns/common/log.h>
#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/FrameBuffer.h>
#include <brayns/engineapi/Renderer.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/manipulators/AbstractManipulator.h>
#include <brayns/parameters/ParametersManager.h>

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

namespace brayns
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

BaseWindow::BaseWindow(Brayns& brayns, const FrameBufferMode frameBufferMode)
    : _brayns(brayns)
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
    const auto motionSpeed = _brayns.getCameraManipulator().getMotionSpeed();
    BRAYNS_INFO("Camera       :" << _brayns.getEngine().getCamera());
    BRAYNS_INFO("Motion speed :" << motionSpeed);

    const auto& rp = _brayns.getParametersManager().getRenderingParameters();
    if (rp.getAccumulationType() == AccumulationType::ai_denoised)
        _frameBufferMode = FrameBufferMode::COLOR_F32;
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

    auto& manipulator = _brayns.getCameraManipulator();

    if (_currModifiers & GLUT_ACTIVE_SHIFT && released)
    {
        const auto& result =
            _brayns.getEngine().getRenderer().pick({pos.x / float(_windowSize.x), 1.f - pos.y / float(_windowSize.y)});
        _brayns.getEngine().getFrameBuffer().clear();
        if (result.hit)
        {
            _brayns.getEngine().getCamera().setTarget(result.pos);
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

    auto& manipulator = _brayns.getCameraManipulator();

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

    // auto& camera = _brayns.getEngine().getCamera();
    // std::cout << "1 - " << camera.getOrientation() << std::endl;
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

    auto& applicationParameters = _brayns.getParametersManager();
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
    const Vector2ui windowSize = _brayns.getParametersManager().getApplicationParameters().getWindowSize();
    if (windowSize != _windowSize)
        glutReshapeWindow(windowSize.x, windowSize.y);

    _timer.start();

    const auto& camera = _brayns.getEngine().getCamera();
    _renderInput.windowSize = windowSize;
    _renderInput.position = camera.getPosition();
    _renderInput.orientation = camera.getOrientation();
    _renderInput.target = camera.getTarget();

    const auto& fb = _brayns.getEngine().getFrameBuffer();
    const auto& rp = _brayns.getParametersManager().getRenderingParameters();
    const auto maxAccumFrames = rp.getMaxAccumFrames();

    if (fb.numAccumFrames() < maxAccumFrames)
        _brayns.commitAndRender(_renderInput, _renderOutput);
    else
        _brayns.commit();

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

    if (_displayHelp)
    {
        auto& keyHandler = _brayns.getKeyboardHandler();
        std::string help;
        for (const auto& value : keyHandler.help())
            help += value + "\n";

        glLogicOp(GL_XOR);
        glEnable(GL_COLOR_LOGIC_OP);
        _renderBitmapString(-0.98f, 0.95f, help);
        glDisable(GL_COLOR_LOGIC_OP);
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

void BaseWindow::setTitle(const char* title)
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
    case 27:
    case 'Q':
#ifdef __APPLE__
        exit(0);
#else
        glutLeaveMainLoop();
#endif
        break;
    default:
        _brayns.getKeyboardHandler().handleKeyboardShortcut(key);
    }

    _brayns.getEngine().commit();
}

void BaseWindow::specialkey(const int key, const Vector2f&)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:
        _brayns.getKeyboardHandler().handle(SpecialKey::LEFT);
        break;
    case GLUT_KEY_RIGHT:
        _brayns.getKeyboardHandler().handle(SpecialKey::RIGHT);
        break;
    case GLUT_KEY_UP:
        _brayns.getKeyboardHandler().handle(SpecialKey::UP);
        break;
    case GLUT_KEY_DOWN:
        _brayns.getKeyboardHandler().handle(SpecialKey::DOWN);
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
    auto& keyHandler = _brayns.getKeyboardHandler();
    keyHandler.registerKeyboardShortcut('z', "Switch between depth and color buffers",
                                        std::bind(&BaseWindow::_toggleFrameBuffer, this));
}

#ifdef __APPLE__
void BaseWindow::_renderBitmapString(const float, const float, const std::string&) {}
#else
void BaseWindow::_renderBitmapString(const float x, const float y, const std::string& text)
{
    glRasterPos3f(x, y, 0.f);
    glutBitmapString(GLUT_BITMAP_8_BY_13, reinterpret_cast<const unsigned char*>(text.c_str()));
    glRasterPos3f(-1.f, -1.f, 0.f);
}
#endif

void BaseWindow::_toggleFrameBuffer()
{
    size_t mode = static_cast<size_t>(_frameBufferMode);
    mode = (mode + 1) % 3;
    auto& engine = _brayns.getEngine();
    const auto& params = engine.getParametersManager().getApplicationParameters();
    const auto& engineName = params.getEngine();
    if (engineName == ENGINE_OSPRAY && mode == static_cast<size_t>(AccumulationType::ai_denoised))
        mode = (mode + 1) % 3;
    _frameBufferMode = static_cast<FrameBufferMode>(mode);

    // Accumulation type
    auto& frameBuffer = engine.getFrameBuffer();
    if (_frameBufferMode == FrameBufferMode::COLOR_F32)
        frameBuffer.setAccumulationType(AccumulationType::ai_denoised);
    else
        frameBuffer.setAccumulationType(AccumulationType::linear);
}
} // namespace brayns
