
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_Utils.h:

Program Listing for File Utils.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_Utils.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/common/Utils.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
    * scientific data from visualization
    *
    * Copyright 2020-2023 Blue BrainProject / EPFL
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
   
   #include <plugin/common/Types.h>
   
   #include <brayns/common/types.h>
   
   namespace bioexplorer
   {
   namespace common
   {
   using namespace brayns;
   using namespace details;
   
   std::string& ltrim(std::string& s);
   
   std::string& rtrim(std::string& s);
   
   std::string& trim(std::string& s);
   
   bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes);
   
   Transformation getSphericalPosition(const Vector3f& position,
                                       const float radius, const size_t occurence,
                                       const size_t occurences,
                                       const RandomizationDetails& randInfo);
   
   Transformation getPlanarPosition(const Vector3f& position, const float size,
                                    const RandomizationDetails& randInfo);
   
   Transformation getCubicPosition(const Vector3f& center, const float size,
                                   const RandomizationDetails& randInfo);
   
   float sinusoide(const float x, const float z);
   
   Transformation getSinosoidalPosition(const Vector3f& center, const float size,
                                        const float amplitude,
                                        const size_t occurence,
                                        const RandomizationDetails& randInfo);
   
   Transformation getFanPosition(const Vector3f& center, const float radius,
                                 const size_t occurence, const size_t occurences,
                                 const RandomizationDetails& randInfo);
   
   Transformation getBezierPosition(const Vector3fs& points, const float scale,
                                    const float t);
   
   Transformation getSphericalToPlanarPosition(
       const Vector3f& center, const float radius, const size_t occurence,
       const size_t occurences, const RandomizationDetails& randInfo,
       const float morphingStep);
   
   void setDefaultTransferFunction(Model& model);
   
   Vector4fs getClippingPlanes(const Scene& scene);
   
   Quaterniond randomQuaternion(const size_t seed);
   } // namespace common
   } // namespace bioexplorer
