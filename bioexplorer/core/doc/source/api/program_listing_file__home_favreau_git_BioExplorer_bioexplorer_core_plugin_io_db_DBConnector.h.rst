
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_db_DBConnector.h:

Program Listing for File DBConnector.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_io_db_DBConnector.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/io/db/DBConnector.h``)

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
   
   #include <brayns/common/types.h>
   
   #include <pqxx/pqxx>
   
   namespace bioexplorer
   {
   namespace io
   {
   namespace db
   {
   using namespace details;
   
   class DBConnector
   {
   public:
       DBConnector(const std::string& connectionString, const std::string& schema);
   
       ~DBConnector();
   
       void clearBricks();
   
       const OOCSceneConfigurationDetails getSceneConfiguration();
   
       std::stringstream getBrick(const int32_t brickId, const uint32_t& version,
                                  uint32_t& nbModels);
   
       const void insertBrick(const int32_t brickId, const uint32_t version,
                              const uint32_t nbModels,
                              const std::stringstream& buffer);
   
   private:
       pqxx::connection _connection;
       std::string _schema;
   };
   
   typedef std::shared_ptr<DBConnector> DBConnectorPtr;
   } // namespace db
   } // namespace io
   } // namespace bioexplorer
