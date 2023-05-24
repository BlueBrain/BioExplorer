/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "camera/Camera.h"

namespace ospray
{
//! Implements a clipped perspective camera
struct OSPRAY_SDK_INTERFACE SphereClippingPerspectiveCamera : public Camera
{
    /*! \brief constructor \internal also creates the ispc-side data structure
     */
    SphereClippingPerspectiveCamera();

    //! \brief common function to help printf-debugging
    /*! Every derived class should override this! */
    virtual std::string toString() const { return "ospray::SphereClippingPerspectiveCamera"; }
    virtual void commit();

public:
    // ------------------------------------------------------------------
    // the parameters we 'parsed' from our parameters
    // ------------------------------------------------------------------
    float fovy;
    float aspect;
    float apertureRadius;
    float focusDistance;
    bool architectural; // orient image plane to be parallel to 'up' and shift
                        // the lens
    bool stereo;
    float interpupillaryDistance; // distance between the two cameras (stereo)

    // Clip planes
    bool enableClippingPlanes{false};
    Ref<Data> clipPlanes;
};

} // namespace ospray
