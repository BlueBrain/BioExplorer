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
/**
 * @brief Structure defining the RGB color of atoms according to the JMol Scheme
 *
 */
typedef struct
{
    short r, g, b;
} RGBColorDetails;
typedef std::map<std::string, RGBColorDetails> RGBColorDetailsMap;

/**
 * @brief Structure defining the entry point response of the remote API
 *
 */
struct Response
{
    /** Status of the response */
    bool status{true};
    /** Contents of the response (optional) */
    std::string contents;
};

/**
 * @brief Structure defining the plugin general settings
 *
 */
typedef struct
{
    bool modelVisibilityOnCreation;
    std::string offFolder;
    bool loggingEnabled;
} GeneralSettingsDetails;

/**
 * @brief Model position randomization types
 *
 */
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

/**
 * @brief Shapes that can be used to enroll RNA into the virus capsid
 *
 */
enum class RNAShape
{
    /** Trefoil knot */
    trefoilKnot = 0,
    /** Torus */
    torus = 1,
    /** Star */
    star = 2,
    /** Spring */
    spring = 3,
    /** Heart (all we need is love) */
    heart = 4,
    /** Thing (weird shape) */
    thing = 5,
    /** Moebius knot */
    moebius = 6
};

/**
 * @brief Assembly representation
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string name;
    /** Position of the assembly in the scene */
    std::vector<float> position;
    /** rotation of the assembly in the scene */
    std::vector<float> rotation;
    /** Clipping planes applied to the loading of elements of the assembly */
    std::vector<float> clippingPlanes;
} AssemblyDetails;

/**
 * @brief Structure defining transformations to apply to assembly elements
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the element in the assembly to which transformations should be
     * applied */
    std::string name;
    /** List of transformations */
    std::vector<float> transformations;
} AssemblyTransformationsDetails;

/**
 * @brief Protein representation (atoms, atoms and sticks, etc)
 *
 */
enum class ProteinRepresentation
{
    /** Atoms only */
    atoms = 0,
    /** Atoms and sticks */
    atoms_and_sticks = 1,
    /** Protein contours */
    contour = 2,
    /** Protein surface computed using metaballs */
    surface = 3,
    /** Protein surface computed using union of balls */
    union_of_balls = 4,
    /** Debug mode, usually showing size and rotation of the protein */
    debug = 5
};

/**
 * @brief Assembly shapes
 *
 */
enum class AssemblyShape
{
    /** Spherical */
    spherical = 0,
    /** Planar */
    planar = 1,
    /** Sinusoidal */
    sinusoidal = 2,
    /** Cubic */
    cubic = 3,
    /** Fan */
    fan = 4,
    /** Bezier (experimental) */
    bezier = 5,
    /** Sphere to plane */
    spherical_to_planar = 6
};

/**
 * @brief A Membrane is a shaped assembly of phospholipids
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** String containing a PDB representation of the 1st protein */
    std::string content1;
    /** String containing a PDB representation of the 2nd optional protein
     */
    std::string content2;
    /** String containing a PDB representation of the 3rd optional protein
     */
    std::string content3;
    /** String containing a PDB representation of the 4th optional protein
     */
    std::string content4;
    /** Shape of the assembly containing the membrane */
    AssemblyShape shape;
    /** Parameters of the assembly shape */
    std::vector<float> assemblyParams;
    /** Multiplier applied to the radius of the protein atoms */
    float atomRadiusMultiplier;
    /** Enable the loading of protein bonds */
    bool loadBonds;
    /** Enable the loading of non polymer chemicals */
    bool loadNonPolymerChemicals;
    /** Defines the representation of the protein (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation;
    /** Identifiers of chains to be loaded */
    std::vector<size_t> chainIds;
    /** Recenters the protein  */
    bool recenter;
    /** Number of protein occurences to be added to the assembly */
    size_t occurrences;
    /** Seed for position randomization */
    size_t randomSeed;
    /** Type of randomisation for the elements of the assembly */
    PositionRandomizationType positionRandomizationType;
    /** Relative rotation of the protein in the assembly */
    std::vector<float> rotation;
} MembraneDetails;

// Protein
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** String containing a PDB representation of the protein */
    std::string contents;
    /** Shape of the assembly containing the protein */
    AssemblyShape shape;
    /** Parameters of the assembly shape */
    std::vector<float> assemblyParams;
    /** Multiplier applied to the radius of the protein atoms */
    float atomRadiusMultiplier;
    /** Enable the loading of protein bonds */
    bool loadBonds;
    /** Enable the loading of non polymer chemicals */
    bool loadNonPolymerChemicals;
    /** Enable the loading of hydrogen atoms */
    bool loadHydrogen;
    /** Defines the representation of the protein (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation;
    /** Identifiers of chains to be loaded */
    std::vector<size_t> chainIds;
    /** Recenters the protein  */
    bool recenter;
    /** Number of protein occurences to be added to the assembly */
    size_t occurrences;
    /** Indices of protein occurences in the assembly for which proteins are
     * added */
    std::vector<size_t> allowedOccurrences;
    /** Seed for position randomization */
    size_t randomSeed;
    /** Type of randomisation for the elements of the assembly */
    PositionRandomizationType positionRandomizationType;
    /** Relative position of the protein in the assembly */
    std::vector<float> position;
    /** Relative rotation of the protein in the assembly */
    std::vector<float> rotation;
} ProteinDetails;

/**
 * @brief Data structure describing the glycans
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the glycans in the assembly */
    std::string name;
    /** String containing a PDB representation of the glycans */
    std::string contents;
    /** Name of the protein on which glycans are added */
    std::string proteinName;
    /** Multiplier applied to the radius of the protein atoms */
    float atomRadiusMultiplier;
    /** Enable the loading of protein bonds */
    bool loadBonds;
    /** Defines the representation of the protein (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation;
    /** Recenters the protein  */
    bool recenter;
    /** Identifiers of chains to be loaded */
    std::vector<size_t> chainIds;
    /** List of sites on which glycans can be added */
    std::vector<size_t> siteIndices;
    /** Relative rotation of the glycans on the protein */
    std::vector<float> rotation;
} SugarsDetails;

/**
 * @brief Data structure describing a membrane based on the shape of a mesh
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the mesh of the assembly */
    std::string name;
    /** String containing an OBJ representation of the mesh */
    std::string meshContents;
    /** String containing an PDB representation of the protein */
    std::string proteinContents;
    /** Recenters the protein  */
    bool recenter;
    /** Density of proteins in surface of the mesh */
    float density;
    /** Fixed offset for the position of the protein above the surface of the
     * mesh*/
    float surfaceFixedOffset;
    /** ariable (randomized) offset for the position of the protein above the
     * surface of the mesh*/
    float surfaceVariableOffset;
    /** Multiplier applied to atom radius */
    float atomRadiusMultiplier;
    /** Representation of the protein (Atoms, atoms and sticks, etc) */
    ProteinRepresentation representation;
    /** Seed for randomization of the variable offset */
    size_t randomSeed;
    /** Relative position of the mesh in the assembly */
    std::vector<float> position;
    /** Relative rotation of the mesh in the assembly */
    std::vector<float> rotation;
    /** Scale of the mesh */
    std::vector<float> scale;
} MeshBasedMembraneDetails;

/**
 * @brief RNA sequence descriptor
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the RNA sequence in the assembly */
    std::string name;
    /** A string containing the list of codons */
    std::string contents;
    /** A given shape */
    RNAShape shape;
    /** Assembly parameters (size) */
    std::vector<float> assemblyParams;
    /** Range of values used to compute the shape */
    std::vector<float> range;
    /** Parameters used to compute the shape */
    std::vector<float> params;
    /** Relative position of the RNA sequence in the assembly */
    std::vector<float> position;
} RNASequenceDetails;

/**
 * @brief Structure defining a selection of amino acids on a protein of
 * an assembly. The selection is defined as a string
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** String containing the amino acid sequence to select */
    std::string sequence;
} AminoAcidSequenceAsStringDetails;

/**
 * @brief Structure defining a selection of amino acids on a protein of an
 * assembly. The selection is defined as a range of indices
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** List of tuples of 2 integers defining indices in the sequence of amino
     * acid */
    std::vector<size_t> ranges;
} AminoAcidSequenceAsRangesDetails;

typedef struct
{
    std::string assemblyName;
    std::string name;
} AminoAcidInformationDetails;

/**
 * @brief Structure used to set an amino acid in protein sequences
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein */
    std::string name;
    /** Index of the amino acid in the sequence */
    size_t index;
    /** Amino acid short name */
    std::string aminoAcidShortName;
    /** List of chains in which the amino acid is set */
    std::vector<size_t> chainIds;
} AminoAcidDetails;

/**
 * @brief Defines the parameters needed when adding 3D grid in the scene
 *
 */
typedef struct
{
    /** Minimum value on the axis */
    float minValue;
    /** Maximum value on the axis */
    float maxValue;
    /** Interval between lines of the grid */
    float steps;
    /** Radius of the lines */
    float radius;
    /** Opacity of the grid */
    float planeOpacity;
    /** Defines if axes should be shown */
    bool showAxis;
    /** Defines if planes should be shown */
    bool showPlanes;
    /** Defines if full grid should be shown */
    bool showFullGrid;
    /** Defines if the RGB color scheme shoudl be applied to axis */
    bool useColors;
    /** Position of the grid in the scene */
    std::vector<float> position;
} AddGridDetails;

/**
 * @brief Color schemes that can be applied to proteins
 *
 */
enum class ColorScheme
{
    /** All atoms use the same color */
    none = 0,
    /** Colored by atom according to the Pymol color scheme */
    atoms = 1,
    /** Colored by chain */
    chains = 2,
    /** Colored by residue */
    residues = 3,
    /** Colored by sequence of amino acids */
    amino_acid_sequence = 4,
    /** Colored by glysolysation site */
    glycosylation_site = 5,
    /** Colored by functional region */
    region = 6
};

/**
 * @brief Defines the color scheme to apply to a protein
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** Color scheme **/
    ColorScheme colorScheme;
    /** Palette of colors (RGB values) */
    std::vector<float> palette;
    /** Ids of protein chains to which the colors scheme is applied */
    std::vector<size_t> chainIds;
} ColorSchemeDetails;

typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** Index of the protein instance */
    size_t instanceIndex;
    /** Position of the protein instance */
    std::vector<float> position;
    /** rotation of the protein instance */
    std::vector<float> rotation;
} ProteinInstanceTransformationDetails;

/**
 * @brief List of material identifiers attached to a Brayns model
 *
 */
typedef struct
{
    /** List of material identifiers */
    std::vector<size_t> ids;
} MaterialIdsDetails;

/**
 * @brief Model identifier
 *
 */
typedef struct
{
    /** Model identifier */
    size_t modelId;
} ModelIdDetails;

/**
 * @brief Structure containing attributes of materials attached to one or
 several Brayns models
 */
typedef struct
{
    /** List of model identifiers */
    std::vector<int32_t> modelIds;
    /** List of material identifiers */
    std::vector<int32_t> materialIds;
    /** List of RGB values for diffuse colors */
    std::vector<float> diffuseColors;
    /** List of RGB values for specular colors */
    std::vector<float> specularColors;
    /** List of values for specular exponents */
    std::vector<float> specularExponents;
    /** List of values for reflection indices */
    std::vector<float> reflectionIndices;
    /** List of values for opacities */
    std::vector<float> opacities;
    /** List of values for refraction indices */
    std::vector<float> refractionIndices;
    /** List of values for light emission */
    std::vector<float> emissions;
    /** List of values for glossiness */
    std::vector<float> glossinesses;
    /** List of values for shading modes */
    std::vector<int32_t> shadingModes;
    /** List of values for user defined parameters */
    std::vector<float> userParameters;
    /** List of values for chameleon mode parameters */
    std::vector<int32_t> chameleonModes;
} MaterialsDetails;

/**
 * @brief Structure containing information about how to build magnetic fields
 * from atom positions and charge
 *
 */
typedef struct
{
    /** Voxel size used to build the Octree acceleration structure */
    float voxelSize;
    /** Density of atoms to consider (Between 0 and 1) */
    float density;
} BuildFieldsDetails;

// IO
typedef struct
{
    size_t modelId;
    std::string filename;
} ModelIdFileAccessDetails;

/**
 * @brief File format for export of atom coordinates, radius and charge
 *
 */
enum class XYZFileFormat
{
    /** Unspecified */
    unspecified = 0,
    /** x, y, z coordinates stored in binary representation (4 byte float) */
    xyz_binary = 1,
    /** x, y, z coordinates and radius stored in binary representation (4 byte
       float) */
    xyzr_binary = 2,
    /** x, y, z coordinates, radius, and charge stored in binary representation
       (4 byte float) */
    xyzrv_binary = 3,
    /** x, y, z coordinates stored in space separated ascii representation. One
       line per atom*/
    xyz_ascii = 4,
    /** x, y, z coordinates and radius stored in space separated ascii
       representation. One line per atom*/
    xyzr_ascii = 5,
    /** x, y, z coordinates, radius, and charge stored in space separated ascii
       representation. One line per atom*/
    xyzrv_ascii = 6
};

/**
 * @brief Structure defining how to export data into a file
 *
 */
typedef struct
{
    std::string filename;
    std::vector<float> lowBounds;
    std::vector<float> highBounds;
    XYZFileFormat fileFormat;
} FileAccessDetails;

/**
 * @brief Structure defining how to export data into a DB
 *
 */
typedef struct
{
    std::string connectionString;
    std::string schema;
    int32_t brickId;
    std::vector<float> lowBounds;
    std::vector<float> highBounds;
} DatabaseAccessDetails;

/**
 * @brief Structure defining how to build a point cloud from the scene
 *
 */
typedef struct
{
    float radius;
} BuildPointCloudDetails;

/**
 * @brief Structure defining how visible models are in the scene
 *
 */
typedef struct
{
    bool visible;
} ModelsVisibilityDetails;

typedef struct
{
    /** Description of the scene **/
    std::string description;
    /** Size of the scene */
    Vector3f sceneSize;
    /** Number of bricks per side of the scene */
    uint32_t nbBricks;
    /** Size of the each brick in the scene */
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

/**
 * @brief Structure containing information about an atom, as stored in a PDB
 * file
 *
 */
typedef struct
{
    /** Atom name */
    std::string name;
    /** Alternate location indicator */
    std::string altLoc;
    /** Residue name */
    std::string resName;
    /** Chain identifier */
    std::string chainId;
    /** Residue sequence number */
    size_t reqSeq;
    /** Code for insertions of residues */
    std::string iCode;
    /** orthogonal angstrom coordinates */
    Vector3f position;
    /** Occupancy */
    float occupancy;
    /** Temperature factor */
    float tempFactor;
    /** Element symbol */
    std::string element;
    /** Charge */
    std::string charge;
    /** Radius */
    float radius;
} Atom;
typedef std::multimap<size_t, Atom, std::less<size_t>> AtomMap;

/**
 * @brief Sequence of residues
 *
 */
typedef struct
{
    /** Number of residues in the chain */
    size_t numRes;
    /** Residue name */
    std::vector<std::string> resNames;
    /** Atom Offset */
    size_t offset;
} ResidueSequence;
typedef std::map<std::string, ResidueSequence> ResidueSequenceMap;

/**
 * @brief Bonds map
 *
 */
typedef std::map<size_t, std::vector<size_t>> BondsMap;

/**
 * @brief Structure containing amino acids long and shot names
 *
 */
typedef struct
{
    /** Long name of the amino acid*/
    std::string name;
    /** Short name of the amino acid*/
    std::string shortName;
} AminoAcid;
typedef std::map<std::string, AminoAcid> AminoAcidMap;

/**
 * @brief Set of residue names
 *
 */
typedef std::set<std::string> Residues;

/**
 * @brief Atom radii in microns
 *
 */
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
