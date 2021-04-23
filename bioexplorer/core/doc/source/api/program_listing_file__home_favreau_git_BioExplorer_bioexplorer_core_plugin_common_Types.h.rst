
.. _program_listing_file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_Types.h:

Program Listing for File Types.h
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_favreau_git_BioExplorer_bioexplorer_core_plugin_common_Types.h>` (``/home/favreau/git/BioExplorer/bioexplorer/core/plugin/common/Types.h``)

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
   
   #include <Defines.h>
   #include <brayns/engineapi/Scene.h>
   
   #include <map>
   #include <set>
   #include <string>
   #include <vector>
   
   namespace bioexplorer
   {
   using namespace brayns;
   
   // Consts
   const float BOND_RADIUS = 0.025f;
   const float DEFAULT_STICK_DISTANCE = 0.175f;
   const brayns::Vector3f UP_VECTOR = {0.f, 0.f, 1.f};
   
   // Metadata
   const std::string METADATA_ASSEMBLY = "Assembly";
   const std::string METADATA_TITLE = "Title";
   const std::string METADATA_HEADER = "Header";
   const std::string METADATA_ATOMS = "Atoms";
   const std::string METADATA_BONDS = "Bonds";
   const std::string METADATA_SIZE = "Size";
   const std::string METADATA_BRICK_ID = "BrickId";
   
   // Command line arguments
   #ifdef USE_PQXX
   const std::string ARG_OOC_ENABLED = "--ooc-enabled";
   const std::string ARG_OOC_DB_HOST = "--ooc-db-host";
   const std::string ARG_OOC_DB_PORT = "--ooc-db-port";
   const std::string ARG_OOC_DB_NAME = "--ooc-db-dbname";
   const std::string ARG_OOC_DB_USER = "--ooc-db-user";
   const std::string ARG_OOC_DB_PASSWORD = "--ooc-db-password";
   const std::string ARG_OOC_DB_SCHEMA = "--ooc-db-schema";
   #else
   const std::string ARG_OOC_BRICKS_FOLDER = "--ooc-bricks-folder";
   #endif
   const std::string ARG_OOC_VISIBLE_BRICKS = "--ooc-visible-bricks";
   const std::string ARG_OOC_UPDATE_FREQUENCY = "--ooc-update-frequency";
   const std::string ARG_OOC_UNLOAD_BRICKS = "--ooc-unload-bricks";
   const std::string ARG_OOC_SHOW_GRID = "--ooc-show-grid";
   const std::string ARG_OOC_NB_BRICKS_PER_CYCLE = "--ooc-nb-bricks-per-cycle";
   
   // Environment variables
   const std::string ENV_ROCKETS_DISABLE_SCENE_BROADCASTING =
       "ROCKETS_DISABLE_SCENE_BROADCASTING";
   
   // Typedefs
   typedef std::map<std::string, std::string> StringMap;
   typedef Vector3f Color;
   typedef std::vector<Color> Palette;
   typedef std::vector<Quaterniond> Quaternions;
   typedef std::vector<float> floats;
   typedef std::vector<Vector3f> Vector3fs;
   typedef std::vector<Vector2ui> Vector2uis;
   typedef std::vector<uint32_t> uint32_ts;
   typedef std::map<std::string, std::string> CommandLineArguments;
   
   namespace details
   {
   typedef struct
   {
       short r, g, b;
   } RGBColorDetails;
   typedef std::map<std::string, RGBColorDetails> RGBColorDetailsMap;
   
   struct Response
   {
       bool status{true};
       std::string contents;
   };
   
   typedef struct
   {
       bool modelVisibilityOnCreation;
       std::string offFolder;
       bool loggingEnabled;
   } GeneralSettingsDetails;
   
   enum class PositionRandomizationType
   {
       circular = 0,
       radial = 1
   };
   
   typedef struct
   {
       uint32_t seed;
       PositionRandomizationType randomizationType;
       uint32_t positionSeed;
       float positionStrength;
       uint32_t rotationSeed;
       float rotationStrength;
   } RandomizationDetails;
   
   enum class RNAShape
   {
       trefoilKnot = 0,
       torus = 1,
       star = 2,
       spring = 3,
       heart = 4,
       thing = 5,
       moebius = 6
   };
   
   typedef struct
   {
       std::string name;
       std::vector<float> position;
       std::vector<float> rotation;
       std::vector<float> clippingPlanes;
   } AssemblyDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::vector<float> transformations;
   } AssemblyTransformationsDetails;
   
   enum class ProteinRepresentation
   {
       atoms = 0,
       atoms_and_sticks = 1,
       contour = 2,
       surface = 3,
       union_of_balls = 4,
       debug = 5
   };
   
   enum class AssemblyShape
   {
       spherical = 0,
       planar = 1,
       sinusoidal = 2,
       cubic = 3,
       fan = 4,
       bezier = 5,
       spherical_to_planar = 6
   };
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::string content1;
       std::string content2;
       std::string content3;
       std::string content4;
       AssemblyShape shape;
       std::vector<float> assemblyParams;
       float atomRadiusMultiplier;
       bool loadBonds;
       bool loadNonPolymerChemicals;
       ProteinRepresentation representation;
       std::vector<size_t> chainIds;
       bool recenter;
       size_t occurrences;
       size_t randomSeed;
       PositionRandomizationType positionRandomizationType;
       std::vector<float> rotation;
   } MembraneDetails;
   
   // Protein
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::string contents;
       AssemblyShape shape;
       std::vector<float> assemblyParams;
       float atomRadiusMultiplier;
       bool loadBonds;
       bool loadNonPolymerChemicals;
       bool loadHydrogen;
       ProteinRepresentation representation;
       std::vector<size_t> chainIds;
       bool recenter;
       size_t occurrences;
       std::vector<size_t> allowedOccurrences;
       size_t randomSeed;
       PositionRandomizationType positionRandomizationType;
       std::vector<float> position;
       std::vector<float> rotation;
   } ProteinDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::string contents;
       std::string proteinName;
       float atomRadiusMultiplier;
       bool loadBonds;
       ProteinRepresentation representation;
       bool recenter;
       std::vector<size_t> chainIds;
       std::vector<size_t> siteIndices;
       std::vector<float> rotation;
   } SugarsDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::string meshContents;
       std::string proteinContents;
       bool recenter;
       float density;
       float surfaceFixedOffset;
       float surfaceVariableOffset;
       float atomRadiusMultiplier;
       ProteinRepresentation representation;
       size_t randomSeed;
       std::vector<float> position;
       std::vector<float> rotation;
       std::vector<float> scale;
   } MeshBasedMembraneDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::string contents;
       RNAShape shape;
       std::vector<float> assemblyParams;
       std::vector<float> range;
       std::vector<float> params;
       std::vector<float> position;
   } RNASequenceDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::string sequence;
   } AminoAcidSequenceAsStringDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       std::vector<size_t> ranges;
   } AminoAcidSequenceAsRangesDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
   } AminoAcidInformationDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       size_t index;
       std::string aminoAcidShortName;
       std::vector<size_t> chainIds;
   } AminoAcidDetails;
   
   typedef struct
   {
       float minValue;
       float maxValue;
       float steps;
       float radius;
       float planeOpacity;
       bool showAxis;
       bool showPlanes;
       bool showFullGrid;
       bool useColors;
       std::vector<float> position;
   } AddGridDetails;
   
   enum class ColorScheme
   {
       none = 0,
       atoms = 1,
       chains = 2,
       residues = 3,
       amino_acid_sequence = 4,
       glycosylation_site = 5,
       region = 6
   };
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       ColorScheme colorScheme;
       std::vector<float> palette;
       std::vector<size_t> chainIds;
   } ColorSchemeDetails;
   
   typedef struct
   {
       std::string assemblyName;
       std::string name;
       size_t instanceIndex;
       std::vector<float> position;
       std::vector<float> rotation;
   } ProteinInstanceTransformationDetails;
   
   typedef struct
   {
       std::vector<size_t> ids;
   } MaterialIdsDetails;
   
   typedef struct
   {
       size_t modelId;
   } ModelIdDetails;
   
   typedef struct
   {
       std::vector<int32_t> modelIds;
       std::vector<int32_t> materialIds;
       std::vector<float> diffuseColors;
       std::vector<float> specularColors;
       std::vector<float> specularExponents;
       std::vector<float> reflectionIndices;
       std::vector<float> opacities;
       std::vector<float> refractionIndices;
       std::vector<float> emissions;
       std::vector<float> glossinesses;
       std::vector<int32_t> shadingModes;
       std::vector<float> userParameters;
       std::vector<int32_t> chameleonModes;
   } MaterialsDetails;
   
   typedef struct
   {
       float voxelSize;
       float density;
   } BuildFieldsDetails;
   
   // IO
   typedef struct
   {
       size_t modelId;
       std::string filename;
   } ModelIdFileAccessDetails;
   
   enum class XYZFileFormat
   {
       unspecified = 0,
       xyz_binary = 1,
       xyzr_binary = 2,
       xyzrv_binary = 3,
       xyz_ascii = 4,
       xyzr_ascii = 5,
       xyzrv_ascii = 6
   };
   
   typedef struct
   {
       std::string filename;
       std::vector<float> lowBounds;
       std::vector<float> highBounds;
       XYZFileFormat fileFormat;
   } FileAccessDetails;
   
   typedef struct
   {
       std::string connectionString;
       std::string schema;
       int32_t brickId;
       std::vector<float> lowBounds;
       std::vector<float> highBounds;
   } DatabaseAccessDetails;
   
   typedef struct
   {
       float radius;
   } BuildPointCloudDetails;
   
   typedef struct
   {
       bool visible;
   } ModelsVisibilityDetails;
   
   typedef struct
   {
       std::string description;
       Vector3f sceneSize;
       uint32_t nbBricks;
       Vector3f brickSize;
   } OOCSceneConfigurationDetails;
   } // namespace details
   
   namespace biology
   {
   class Node;
   typedef std::shared_ptr<Node> NodePtr;
   typedef std::map<std::string, NodePtr> NodeMap;
   
   class Assembly;
   typedef std::shared_ptr<Assembly> AssemblyPtr;
   typedef std::map<std::string, AssemblyPtr> AssemblyMap;
   
   class Membrane;
   typedef std::shared_ptr<Membrane> MembranePtr;
   
   class Protein;
   typedef std::shared_ptr<Protein> ProteinPtr;
   typedef std::map<std::string, ProteinPtr> ProteinMap;
   
   class Glycans;
   typedef std::shared_ptr<Glycans> GlycansPtr;
   typedef std::map<std::string, GlycansPtr> GlycansMap;
   
   class MeshBasedMembrane;
   typedef std::shared_ptr<MeshBasedMembrane> MeshBasedMembranePtr;
   typedef std::map<std::string, MeshBasedMembranePtr> MeshBasedMembraneMap;
   
   class RNASequence;
   typedef std::shared_ptr<RNASequence> RNASequencePtr;
   typedef std::map<std::string, std::string> RNASequenceMap;
   
   typedef struct
   {
       std::string name;
       std::string altLoc;
       std::string resName;
       std::string chainId;
       size_t reqSeq;
       std::string iCode;
       Vector3f position;
       float occupancy;
       float tempFactor;
       std::string element;
       std::string charge;
       float radius;
   } Atom;
   typedef std::multimap<size_t, Atom, std::less<size_t>> AtomMap;
   
   typedef struct
   {
       size_t numRes;
       std::vector<std::string> resNames;
       size_t offset;
   } ResidueSequence;
   typedef std::map<std::string, ResidueSequence> ResidueSequenceMap;
   
   typedef std::map<size_t, std::vector<size_t>> BondsMap;
   
   typedef struct
   {
       std::string name;
       std::string shortName;
   } AminoAcid;
   typedef std::map<std::string, AminoAcid> AminoAcidMap;
   
   typedef std::set<std::string> Residues;
   
   typedef std::map<std::string, float> AtomicRadii;
   
   // Amino acids
   static AminoAcidMap aminoAcidMap = {{".", {".", "."}},
                                       {"ALA", {"Alanine", "A"}},
                                       {"CYS", {"Cysteine", "C"}},
                                       {"ASP", {"Aspartic acid", "D"}},
                                       {"GLU", {"Glutamic acid", "E"}},
                                       {"PHE", {"Phenylalanine", "F"}},
                                       {"GLY", {"Glycine", "G"}},
                                       {"HIS", {"Histidine", "H"}},
                                       {"ILE", {"Isoleucine", "I"}},
                                       {"LYS", {"Lysine", "K"}},
                                       {"LEU", {"Leucine", "L"}},
                                       {"MET", {"Methionine", "M"}},
                                       {"ASN", {"Asparagine", "N"}},
                                       {"HYP", {"Hydroxyproline", "O"}},
                                       {"PRO", {"Proline", "P"}},
                                       {"GLN", {"Glutamine", "Q"}},
                                       {"ARG", {"Arginine", "R"}},
                                       {"SER", {"Serine", "S"}},
                                       {"THR", {"Threonine", "T"}},
                                       {"GLP", {"Pyroglutamatic", "U"}},
                                       {"VAL", {"Valine", "V"}},
                                       {"TRP", {"Tryptophan", "W"}},
                                       {"TYR", {"Tyrosine", "Y"}}};
   
   // Protein color maps
   static details::RGBColorDetailsMap atomColorMap = {
       {"H", {0xDF, 0xDF, 0xDF}},  {"He", {0xD9, 0xFF, 0xFF}},
       {"Li", {0xCC, 0x80, 0xFF}}, {"Be", {0xC2, 0xFF, 0x00}},
       {"B", {0xFF, 0xB5, 0xB5}},  {"C", {0x90, 0x90, 0x90}},
       {"N", {0x30, 0x50, 0xF8}},  {"O", {0xFF, 0x0D, 0x0D}},
       {"F", {0x9E, 0x05, 0x1}},   {"Ne", {0xB3, 0xE3, 0xF5}},
       {"Na", {0xAB, 0x5C, 0xF2}}, {"Mg", {0x8A, 0xFF, 0x00}},
       {"Al", {0xBF, 0xA6, 0xA6}}, {"Si", {0xF0, 0xC8, 0xA0}},
       {"P", {0xFF, 0x80, 0x00}},  {"S", {0xFF, 0xFF, 0x30}},
       {"Cl", {0x1F, 0xF0, 0x1F}}, {"Ar", {0x80, 0xD1, 0xE3}},
       {"K", {0x8F, 0x40, 0xD4}},  {"Ca", {0x3D, 0xFF, 0x00}},
       {"Sc", {0xE6, 0xE6, 0xE6}}, {"Ti", {0xBF, 0xC2, 0xC7}},
       {"V", {0xA6, 0xA6, 0xAB}},  {"Cr", {0x8A, 0x99, 0xC7}},
       {"Mn", {0x9C, 0x7A, 0xC7}}, {"Fe", {0xE0, 0x66, 0x33}},
       {"Co", {0xF0, 0x90, 0xA0}}, {"Ni", {0x50, 0xD0, 0x50}},
       {"Cu", {0xC8, 0x80, 0x33}}, {"Zn", {0x7D, 0x80, 0xB0}},
       {"Ga", {0xC2, 0x8F, 0x8F}}, {"Ge", {0x66, 0x8F, 0x8F}},
       {"As", {0xBD, 0x80, 0xE3}}, {"Se", {0xFF, 0xA1, 0x00}},
       {"Br", {0xA6, 0x29, 0x29}}, {"Kr", {0x5C, 0xB8, 0xD1}},
       {"Rb", {0x70, 0x2E, 0xB0}}, {"Sr", {0x00, 0xFF, 0x00}},
       {"Y", {0x94, 0xFF, 0xFF}},  {"Zr", {0x94, 0xE0, 0xE0}},
       {"Nb", {0x73, 0xC2, 0xC9}}, {"Mo", {0x54, 0xB5, 0xB5}},
       {"Tc", {0x3B, 0x9E, 0x9E}}, {"Ru", {0x24, 0x8F, 0x8F}},
       {"Rh", {0x0A, 0x7D, 0x8C}}, {"Pd", {0x69, 0x85, 0x00}},
       {"Ag", {0xC0, 0xC0, 0xC0}}, {"Cd", {0xFF, 0xD9, 0x8F}},
       {"In", {0xA6, 0x75, 0x73}}, {"Sn", {0x66, 0x80, 0x80}},
       {"Sb", {0x9E, 0x63, 0xB5}}, {"Te", {0xD4, 0x7A, 0x00}},
       {"I", {0x94, 0x00, 0x94}},  {"Xe", {0x42, 0x9E, 0xB0}},
       {"Cs", {0x57, 0x17, 0x8F}}, {"Ba", {0x00, 0xC9, 0x00}},
       {"La", {0x70, 0xD4, 0xFF}}, {"Ce", {0xFF, 0xFF, 0xC7}},
       {"Pr", {0xD9, 0xFF, 0xC7}}, {"Nd", {0xC7, 0xFF, 0xC7}},
       {"Pm", {0xA3, 0xFF, 0xC7}}, {"Sm", {0x8F, 0xFF, 0xC7}},
       {"Eu", {0x61, 0xFF, 0xC7}}, {"Gd", {0x45, 0xFF, 0xC7}},
       {"Tb", {0x30, 0xFF, 0xC7}}, {"Dy", {0x1F, 0xFF, 0xC7}},
       {"Ho", {0x00, 0xFF, 0x9C}}, {"Er", {0x00, 0xE6, 0x75}},
       {"Tm", {0x00, 0xD4, 0x52}}, {"Yb", {0x00, 0xBF, 0x38}},
       {"Lu", {0x00, 0xAB, 0x24}}, {"Hf", {0x4D, 0xC2, 0xFF}},
       {"Ta", {0x4D, 0xA6, 0xFF}}, {"W", {0x21, 0x94, 0xD6}},
       {"Re", {0x26, 0x7D, 0xAB}}, {"Os", {0x26, 0x66, 0x96}},
       {"Ir", {0x17, 0x54, 0x87}}, {"Pt", {0xD0, 0xD0, 0xE0}},
       {"Au", {0xFF, 0xD1, 0x23}}, {"Hg", {0xB8, 0xB8, 0xD0}},
       {"Tl", {0xA6, 0x54, 0x4D}}, {"Pb", {0x57, 0x59, 0x61}},
       {"Bi", {0x9E, 0x4F, 0xB5}}, {"Po", {0xAB, 0x5C, 0x00}},
       {"At", {0x75, 0x4F, 0x45}}, {"Rn", {0x42, 0x82, 0x96}},
       {"Fr", {0x42, 0x00, 0x66}}, {"Ra", {0x00, 0x7D, 0x00}},
       {"Ac", {0x70, 0xAB, 0xFA}}, {"Th", {0x00, 0xBA, 0xFF}},
       {"Pa", {0x00, 0xA1, 0xFF}}, {"U", {0x00, 0x8F, 0xFF}},
       {"Np", {0x00, 0x80, 0xFF}}, {"Pu", {0x00, 0x6B, 0xFF}},
       {"Am", {0x54, 0x5C, 0xF2}}, {"Cm", {0x78, 0x5C, 0xE3}},
       {"Bk", {0x8A, 0x4F, 0xE3}}, {"Cf", {0xA1, 0x36, 0xD4}},
       {"Es", {0xB3, 0x1F, 0xD4}}, {"Fm", {0xB3, 0x1F, 0xBA}},
       {"Md", {0xB3, 0x0D, 0xA6}}, {"No", {0xBD, 0x0D, 0x87}},
       {"Lr", {0xC7, 0x00, 0x66}}, {"Rf", {0xCC, 0x00, 0x59}},
       {"Db", {0xD1, 0x00, 0x4F}}, {"Sg", {0xD9, 0x00, 0x45}},
       {"Bh", {0xE0, 0x00, 0x38}}, {"Hs", {0xE6, 0x00, 0x2E}},
       {"Mt", {0xEB, 0x00, 0x26}}, {"none", {0xFF, 0xFF, 0xFF}},
       {"O1", {0xFF, 0x0D, 0x0D}}, {"selection", {0xFF, 0x00, 0x00}}};
   } // namespace biology
   
   namespace io
   {
   // Out of core brick manager
   class OOCManager;
   typedef std::shared_ptr<OOCManager> OOCManagerPtr;
   } // namespace io
   
   } // namespace bioexplorer
