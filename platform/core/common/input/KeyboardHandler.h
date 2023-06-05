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
