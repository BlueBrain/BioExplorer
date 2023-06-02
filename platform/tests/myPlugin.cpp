/* Copyright (c) 2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel.Nachbaur@epfl.ch
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "../plugins/Rockets/jsonSerialization.h"

#include <platform/core/common/ActionInterface.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>
#include <platform/core/pluginapi/Plugin.h>

#include "BasicRenderer_ispc.h"
#include <platform/engines/ospray/ispc/render/AdvancedMaterial.h>
#include <platform/engines/ospray/ispc/render/utils/AbstractRenderer.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using Vec2 = std::array<unsigned, 2>;
const Vec2 vecVal{{1, 1}};

class MyPlugin : public core::ExtensionPlugin
{
public:
    MyPlugin(const int argc, const char** argv)
    {
        if (argc > 0)
        {
            std::cout << "Creating plugin with arguments:";
            for (int i = 0; i < argc; i++)
                std::cout << " " << std::string(argv[i]);
            std::cout);
        }
    }

    void init() final
    {
        auto actions = _api->getActionInterface();
        REQUIRE(actions);

        // test property map for actions
        actions->registerNotification(core::RpcDescription{"notify", "A notification with no params"},
                                      [&] { ++numCalls; });

        core::PropertyMap input;
        input.setProperty({"value", 0});
        actions->registerNotification(core::RpcParameterDescription{"notify-param",
                                                                      "A notification with property map", "param",
                                                                      "a beautiful input param"},
                                      input, [&](const core::PropertyMap& prop) {
                                          if (prop.hasProperty("value"))
                                              CHECK_EQ(prop.getProperty<int>("value"), 42);
                                          else
                                              ++numFails;
                                          ++numCalls;
                                      });

        core::PropertyMap output;
        output.setProperty({"result", true});
        actions->registerRequest(core::RpcDescription{"request", "A request returning a property map"}, output,
                                 [&, output = output] {
                                     ++numCalls;
                                     return output;
                                 });

        actions->registerRequest(core::RpcParameterDescription{"request-param",
                                                                 "A request with a param and returning a property map",
                                                                 "param", "another nice input param"},
                                 input, output, [&, output = output](const core::PropertyMap& prop) {
                                     ++numCalls;
                                     auto val = prop.getProperty<int>("value");
                                     CHECK_EQ(val, 42);
                                     return output;
                                 });

        // test "arbitrary" objects for actions
        actions->registerNotification("hello", [&] { ++numCalls; });
        actions->registerNotification<Vec2>("foo", [&](const Vec2& vec) {
            ++numCalls;
            CHECK(vec == vecVal);
        });

        actions->registerRequest<std::string>("who", [&] {
            ++numCalls;
            return "me";
        });
        actions->registerRequest<Vec2, Vec2>("echo", [&](const Vec2& vec) {
            ++numCalls;
            return vec;
        });

        // test properties from custom renderer
        core::PropertyMap props;
        props.setProperty({"awesome", int32_t(42), 0, 50, {"Best property", "Because it's the best"}});
        _api->getRenderer().setProperties("myrenderer", props);
    }

    ~MyPlugin()
    {
        REQUIRE_EQ(numCalls, 10);
        REQUIRE_EQ(numFails, 1);
    }
    size_t numCalls{0};
    size_t numFails{0};
};

class MyRenderer : public core::AbstractRenderer
{
public:
    MyRenderer() { ispcEquivalent = ispc::BasicRenderer_create(this); }
    void commit() final
    {
        AbstractRenderer::commit();
        ispc::BasicRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _timestamp, spp, _lightPtr,
                                _lightArray.size());
    }
};

OSP_REGISTER_RENDERER(MyRenderer, myrenderer);
OSP_REGISTER_MATERIAL(myrenderer, core::AdvancedMaterial, default);

extern "C" core::ExtensionPlugin* brayns_plugin_create(int argc, const char** argv)
{
    return new MyPlugin(argc, argv);
}
