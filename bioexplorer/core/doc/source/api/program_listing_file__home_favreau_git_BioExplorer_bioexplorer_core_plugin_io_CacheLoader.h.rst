
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_CacheLoader.h:

Program Listing for File CacheLoader.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_CacheLoader.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/io/CacheLoader.h``)

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
   
   #include <plugin/io/db/DBConnector.h>
   
   #include <brayns/parameters/GeometryParameters.h>
   
   using namespace brayns;
   
   namespace bioexplorer
   {
   namespace io
   {
   using namespace db;
   
   const int32_t UNDEFINED_BOX_ID = std::numeric_limits<int32_t>::max();
   
   class CacheLoader : public Loader
   {
   public:
       CacheLoader(Scene& scene, PropertyMap&& loaderParams = {});
   
       std::string getName() const final;
   
       std::vector<std::string> getSupportedExtensions() const final;
   
       bool isSupported(const std::string& filename,
                        const std::string& extension) const final;
   
       static PropertyMap getCLIProperties();
   
       PropertyMap getProperties() const final;
   
       ModelDescriptorPtr importFromBlob(
           Blob&& blob, const LoaderProgress& callback,
           const PropertyMap& properties) const final;
   
       ModelDescriptorPtr importFromFile(
           const std::string& filename, const LoaderProgress& callback,
           const PropertyMap& properties) const final;
   
       std::vector<ModelDescriptorPtr> importModelsFromFile(
           const std::string& filename, const int32_t brickId = UNDEFINED_BOX_ID,
           const LoaderProgress& callback = LoaderProgress(),
           const PropertyMap& properties = PropertyMap()) const;
   
       void exportToFile(const std::string& filename, const Boxd& bounds) const;
   
   #ifdef USE_PQXX
   
       std::vector<ModelDescriptorPtr> importBrickFromDB(
           DBConnector& connector, const int32_t brickId) const;
   
       void exportBrickToDB(DBConnector& connector, const int32_t brickId,
                            const Boxd& bounds) const;
   #endif
   
       void exportToXYZ(const std::string& filename,
                        const XYZFileFormat format) const;
   
   private:
       std::string _readString(std::stringstream& f) const;
   
       ModelDescriptorPtr _importModel(std::stringstream& buffer,
                                       const int32_t brickId) const;
   
       bool _exportModel(const ModelDescriptorPtr modelDescriptor,
                         std::stringstream& buffer, const Boxd& bounds) const;
   
       PropertyMap _defaults;
   };
   } // namespace io
   } // namespace bioexplorer
