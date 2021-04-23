
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Node.h:

Program Listing for File Node.h
===============================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_biology_Node.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/biology/Node.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
    * scientific data from visualization
    *
    * Copyright 2020-2021 Blue BrainProject / EPFL
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
   
   namespace bioexplorer
   {
   namespace biology
   {
   using namespace brayns;
   using namespace details;
   
   class Node
   {
   public:
       Node();
   
       virtual ~Node();
   
       const ModelDescriptorPtr getModelDescriptor() const
       {
           return _modelDescriptor;
       }
   
   protected:
       ModelDescriptorPtr _modelDescriptor{nullptr};
       uint32_t _uuid;
   };
   } // namespace biology
   } // namespace bioexplorer
