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

#include <platform/core/common/Types.h>

#include <functional>

namespace core
{
struct ShortcutInformation
{
    std::string description;
    std::function<void()> functor;
};

enum class SpecialKey
{
    LEFT,
    RIGHT,
    UP,
    DOWN
};

/**
@brief The KeyboardHandler class manages keyboard shortcuts and special keys
*/
class KeyboardHandler
{
public:
    /**
    @brief Registers a keyboard shortcut
    @param key The unsigned char representing the key to be pressed
    @param description A string description of the shortcut
    @param functor A void function to be called when the shortcut is triggered
    */
    void registerKeyboardShortcut(const unsigned char key, const std::string& description,
                                  std::function<void()> functor);

    /**
    @brief Unregisters a keyboard shortcut
    @param key The unsigned char representing the key of the shortcut
    */
    void unregisterKeyboardShortcut(const unsigned char key);

    /**
    @brief Handles a keyboard shortcut
    @param key The unsigned char representing the key of the shortcut
    */
    void handleKeyboardShortcut(const unsigned char key);

    /**
    @brief Registers a special key
    @param key The SpecialKey enum value representing the special key
    @param description A string description of the special key
    @param functor A void function to be called when the special key is triggered
    */
    void registerSpecialKey(const SpecialKey key, const std::string& description, std::function<void()> functor);

    /**
    @brief Unregisters a special key
    @param key The SpecialKey enum value representing the special key
    */
    void unregisterSpecialKey(const SpecialKey key);

    /**
    @brief Handles a special key
    @param key The SpecialKey enum value representing the special key
    */
    void handle(const SpecialKey key);

    /**
    @brief Returns a vector of help string descriptions for all registered keyboard shortcuts and special keys
    @return A const reference to a vector of help string descriptions
    */
    const std::vector<std::string>& help() const;

    /**
    @brief Returns the description of a specific keyboard shortcut
    @param key The unsigned char representing the key of the shortcut
    @return A const string reference to the description of the keyboard shortcut
    */
    const std::string getKeyboardShortcutDescription(const unsigned char key);

private:
    /**
    @brief Builds the vector of help string descriptions for all registered keyboard shortcuts and special keys
    */
    void _buildHelp();

    std::map<unsigned char, ShortcutInformation> _registeredShortcuts; /** A map of registered keyboard shortcuts */
    std::map<SpecialKey, ShortcutInformation> _registeredSpecialKeys;  /** A map of registered special keys */
    std::vector<std::string> _helpStrings; /** A vector of help string descriptions for all registered shortcuts */
};
} // namespace core
