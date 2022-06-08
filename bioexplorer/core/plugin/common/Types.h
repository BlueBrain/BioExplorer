/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <brayns/common/geometry/SDFGeometry.h>
#include <brayns/engineapi/Scene.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bioexplorer
{
using namespace brayns;

// Consts
const Vector3d UP_VECTOR = {0.0, 1.0, 0.0};

const std::string CONTENTS_DELIMITER = "||||";

// Metadata
const std::string METADATA_ASSEMBLY = "Assembly";
const std::string METADATA_PDB_ID = "PDBId";
const std::string METADATA_HEADER = "Header";
const std::string METADATA_ATOMS = "Atoms";
const std::string METADATA_BONDS = "Bonds";
const std::string METADATA_SIZE = "Size";
const std::string METADATA_BRICK_ID = "BrickId";

// Command line arguments
const std::string ARG_DB_HOST = "--db-host";
const std::string ARG_DB_PORT = "--db-port";
const std::string ARG_DB_NAME = "--db-name";
const std::string ARG_DB_USER = "--db-user";
const std::string ARG_DB_PASSWORD = "--db-password";

const std::string ARG_OOC_ENABLED = "--ooc-enabled";
const std::string ARG_OOC_VISIBLE_BRICKS = "--ooc-visible-bricks";
const std::string ARG_OOC_UPDATE_FREQUENCY = "--ooc-update-frequency";
const std::string ARG_OOC_UNLOAD_BRICKS = "--ooc-unload-bricks";
const std::string ARG_OOC_SHOW_GRID = "--ooc-show-grid";
const std::string ARG_OOC_NB_BRICKS_PER_CYCLE = "--ooc-nb-bricks-per-cycle";

// Environment variables
const std::string ENV_ROCKETS_DISABLE_SCENE_BROADCASTING =
    "ROCKETS_DISABLE_SCENE_BROADCASTING";

// Typedefs
using StringMap = std::map<std::string, std::string>;
using Color = Vector3d;
using Palette = std::vector<Color>;
using Quaternions = std::vector<Quaterniond>;
using bools = std::vector<bool>;
using doubles = std::vector<double>;
using strings = std::vector<std::string>;
using Vector3ds = std::vector<Vector3d>;
using Vector4ds = std::vector<Vector4d>;
using Vector2uis = std::vector<Vector2ui>;
using Vector3uis = std::vector<Vector3ui>;
using uint32_ts = std::vector<uint32_t>;
using uint64_ts = std::vector<uint64_t>;
using CommandLineArguments = std::map<std::string, std::string>;
using Transformations = std::vector<Transformation>;

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
using RGBColorDetailsMap = std::map<std::string, RGBColorDetails>;

/**
 * @brief Structure defining the entry point response of the remote API
 *
 */
typedef struct
{
    /** Status of the response */
    bool status{true};
    /** Contents of the response (optional) */
    std::string contents;
} Response;

/**
 * @brief Structure defining on which instance of a model the camera should
 * focus on
 *
 */
typedef struct
{
    /** Model identifier */
    size_t modelId;
    /** Instance identifier */
    size_t instanceId;
    /** camera direction */
    doubles direction;
    /** Distance to the instance */
    double distance;
} FocusOnDetails;

/**
 * @brief Structure defining the plugin general settings
 *
 */
typedef struct
{
    bool modelVisibilityOnCreation;
    std::string meshFolder;
    uint32_t loggingLevel;
    bool v1Compatibility;
} GeneralSettingsDetails;

typedef struct
{
    uint32_t seed{0};
    uint32_t positionSeed{0};
    double positionStrength{0.f};
    uint32_t rotationSeed{0};
    double rotationStrength{0.f};
    double morphingStep{0.f};
} AnimationDetails;

/**
 * @brief Assembly shapes
 *
 */
enum class AssemblyShape
{
    /** Point */
    point = 0,
    /** Empty sphere */
    empty_sphere = 1,
    /** Planar */
    plane = 2,
    /** Sinusoidal */
    sinusoid = 3,
    /** Cubic */
    cube = 4,
    /** Fan */
    fan = 5,
    /** Bezier (experimental) */
    bezier = 6,
    /** mesh-based */
    mesh = 7,
    /** Helix */
    helix = 8,
    /** Filled sphere */
    filled_sphere = 9,
    /** Spherical cell diffusion */
    spherical_cell_diffusion = 10
};

/**
 * @brief Shapes that can be used to enroll RNA into the virus capsid
 *
 */
enum class RNAShapeType
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
 * @brief Object description in the 3D scene
 *
 */
typedef struct
{
    bool hit;
    std::string assemblyName;
    std::string proteinName;
    size_t modelId;
    size_t instanceId;
    doubles position;
} ProteinInspectionDetails;

typedef struct
{
    doubles origin;
    doubles direction;
} InspectionDetails;

/**
 * @brief Assembly representation
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string name;
    /** Shape of the assembly containing the parametric membrane */
    AssemblyShape shape;
    /** Shape parameters */
    doubles shapeParams;
    /** Contents of the mesh for mesh-based shapes */
    std::string shapeMeshContents;
    /** Position of the assembly in the scene */
    doubles position;
    /** rotation of the assembly in the scene */
    doubles rotation;
    /** Clipping planes applied to the loading of elements of the assembly */
    doubles clippingPlanes;
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
    doubles transformations;
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
    debug = 5,
    /** Precomputed OBJ meshes */
    mesh = 6
};

/**
 * @brief A membrane is a shaped assembly of phospholipids
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the lipid in the assembly */
    std::string name;
    /** String containing a list of PDB ids for the lipids, delimited by
     * PDB_CONTENTS_DELIMITER */
    std::string lipidPDBIds;
    /** String containing a list of PDB description for the lipids, delimited by
     * PDB_CONTENTS_DELIMITER */
    std::string lipidContents;
    /** Relative rotation of the lipid in the membrane */
    doubles lipidRotation;
    /** Lipids density  */
    double lipidDensity;
    /** Multiplier applied to the radius of the lipid atoms */
    double atomRadiusMultiplier;
    /** Enable the loading of lipid bonds */
    bool loadBonds;
    /** Enable the loading of non polymer chemicals */
    bool loadNonPolymerChemicals;
    /** Defines the representation of the lipid (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation;
    /** Identifiers of chains to be loaded */
    size_ts chainIds;
    /** Recenters the lipid  */
    bool recenter;
    /** Extra optional parameters for positioning on the molecule */
    doubles animationParams;
} MembraneDetails;

// Protein
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** PDB if of the protein */
    std::string pdbId;
    /** String containing a PDB representation of the protein */
    std::string contents;
    /** Multiplier applied to the radius of the protein atoms */
    double atomRadiusMultiplier{1.f};
    /** Enable the loading of protein bonds */
    bool loadBonds{false};
    /** Enable the loading of non polymer chemicals */
    bool loadNonPolymerChemicals{false};
    /** Enable the loading of hydrogen atoms */
    bool loadHydrogen{false};
    /** Defines the representation of the protein (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation{
        ProteinRepresentation::atoms_and_sticks};
    /** Identifiers of chains to be loaded */
    size_ts chainIds;
    /** Recenters the protein  */
    bool recenter{false};
    /** Number of protein occurrences to be added to the assembly */
    size_t occurrences{1};
    /** Indices of protein occurrences in the assembly for which proteins are
     * added */
    size_ts allowedOccurrences;
    /** Trans-membrane parameters */
    doubles transmembraneParams;
    /** Extra optional parameters for positioning on the molecule */
    doubles animationParams;
    /** Relative position of the protein in the assembly */
    doubles position;
    /** Relative rotation of the protein in the assembly */
    doubles rotation;
    /** List of assembly names used to constrain the placement of the protein.
     * If the assembly name is prefixed by a +, proteins are not allowed inside
     * the spedified assembly. If the name is prefixed by a -, proteins are not
     * allowed outside of the assembly */
    std::string constraints;
} ProteinDetails;

/**
 * @brief Data structure describing the sugar
 *
 */
typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the sugar in the assembly */
    std::string name;
    /** String containing the PDB Id of the sugar */
    std::string pdbId;
    /** String containing a PDB representation of the sugar */
    std::string contents;
    /** Name of the protein on which sugar are added */
    std::string proteinName;
    /** Multiplier applied to the radius of the molecule atoms */
    double atomRadiusMultiplier;
    /** Enable the loading of molecule bonds */
    bool loadBonds;
    /** Defines the representation of the molecule (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation;
    /** Recenters the protein  */
    bool recenter;
    /** Identifiers of chains to be loaded */
    size_ts chainIds;
    /** List of sites on which sugar can be added */
    size_ts siteIndices;
    /** Relative rotation of the sugar on the molecule */
    doubles rotation;
    /** Extra optional parameters for positioning on the molecule */
    doubles animationParams;
} SugarDetails;

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
    /** String containing the PDB id of the N protein */
    std::string pdbId;
    /** A string containing the list of codons */
    std::string contents;
    /** A string containing an PDB representation of the N protein */
    std::string proteinContents;
    /** A given shape */
    RNAShapeType shape;
    /** Shape radius */
    doubles shapeParams;
    /** Range of values used to compute the shape */
    doubles valuesRange;
    /** Parameters used to compute the shape */
    doubles curveParams;
    /** Multiplier applied to the radius of the molecule atoms */
    double atomRadiusMultiplier;
    /** Defines the representation of the molecule (Atoms, atoms and sticks,
     * surface, etc) */
    ProteinRepresentation representation;
    /** Animation params */
    doubles animationParams;
    /** Relative position of the RNA sequence in the assembly */
    doubles position;
    /** Relative rotation of the RNA sequence in the assembly */
    doubles rotation;
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
    /** List of tuples of 2 integers defining indices in the sequence of
     * amino acid */
    size_ts ranges;
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
    size_ts chainIds;
} AminoAcidDetails;

/**
 * @brief An enzyme reaction
 *
 */
typedef struct
{
    /** Name of the assembly that owns the enzyme reaction */
    std::string assemblyName;
    /** Name of the enzyme reaction in the assembly */
    std::string name;
    /** String containing a list of PDB description for the enzyme protein */
    std::string enzymeName;
    /** String containing a list substrate names */
    std::string substrateNames;
    /** String containing a list of product names */
    std::string productNames;
} EnzymeReactionDetails;

/**
 * @brief Progress of an enzyme reaction for a given instance
 *
 */
typedef struct
{
    /** Name of the assembly that owns the enzyme reaction */
    std::string assemblyName;
    /** Name of the enzyme reaction in the assembly */
    std::string name;
    /** Instance of the substrate molecule */
    size_t instanceId;
    /** Double containing the progress of the reaction (0..1) */
    double progress;
} EnzymeReactionProgressDetails;

/**
 * @brief Defines the parameters needed when adding 3D grid in the scene
 *
 */
typedef struct
{
    /** Minimum value on the axis */
    double minValue;
    /** Maximum value on the axis */
    double maxValue;
    /** Interval between lines of the grid */
    double steps;
    /** Radius of the lines */
    double radius;
    /** Opacity of the grid */
    double planeOpacity;
    /** Defines if axes should be shown */
    bool showAxis;
    /** Defines if planes should be shown */
    bool showPlanes;
    /** Defines if full grid should be shown */
    bool showFullGrid;
    /** Defines if the RGB color scheme shoudl be applied to axis */
    bool useColors;
    /** Position of the grid in the scene */
    doubles position;
} AddGridDetails;

/**
 * @brief Defines the parameters needed when adding sphere to the scene
 *
 */
typedef struct
{
    /** Name of the sphere */
    std::string name;
    /** Position of the sphere in the scene */
    doubles position;
    /** Radius of the sphere */
    double radius;
    /** RGB Color of the sphere */
    doubles color;
    /** Opacity */
    double opacity;
} AddSphereDetails;

/**
 * @brief Defines the parameters needed when adding cone to the scene
 *
 */
typedef struct
{
    /** Name of the cone */
    std::string name;
    /** Origin of the cone in the scene */
    doubles origin;
    /** Target of the cone in the scene */
    doubles target;
    /** Origin radius of the cone */
    double originRadius;
    /** Target radius of the cone */
    double targetRadius;
    /** RGB Color of the cone */
    doubles color;
    /** Opacity */
    double opacity;
} AddConeDetails;

/**
 * @brief Defines the parameters needed when adding 3D sphere to the scene
 *
 */
typedef struct
{
    /** Name of the bounding box */
    std::string name;
    /** Position of the bottom left corner in the scene */
    doubles bottomLeft;
    /** Position of the top right corner in the scene */
    doubles topRight;
    /** Radius of the borders */
    double radius;
    /** RGB Color of the sphere */
    doubles color;
} AddBoundingBoxDetails;

/**
 * @brief Color schemes that can be applied to proteins
 *
 */
enum class ProteinColorScheme
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
    ProteinColorScheme colorScheme;
    /** Palette of colors (RGB values) */
    doubles palette;
    /** Ids of protein chains to which the colors scheme is applied */
    size_ts chainIds;
} ProteinColorSchemeDetails;

typedef struct
{
    /** Name of the assembly */
    std::string assemblyName;
    /** Name of the protein in the assembly */
    std::string name;
    /** Index of the protein instance */
    size_t instanceIndex;
    /** Position of the protein instance */
    doubles position;
    /** rotation of the protein instance */
    doubles rotation;
} ProteinInstanceTransformationDetails;

/**
 * @brief List of identifiers
 *
 */
typedef struct
{
    /** List of identifiers */
    size_ts ids;
} IdsDetails;

/**
 * @brief Model name
 *
 */
typedef struct
{
    /** Element name */
    std::string name;
} NameDetails;

/**
 * @brief Model identifier
 *
 */
typedef struct
{
    /** Model identifier */
    size_t modelId;
    /** Maximum number of instances that can be processed */
    size_t maxNbInstances;
} ModelIdDetails;

/**
 * @brief Structure containing attributes of materials attached to one or
 several Brayns models
 */
typedef struct
{
    /** List of model identifiers */
    int32_ts modelIds;
    /** List of material identifiers */
    int32_ts materialIds;
    /** List of RGB values for diffuse colors */
    doubles diffuseColors;
    /** List of RGB values for specular colors */
    doubles specularColors;
    /** List of values for specular exponents */
    doubles specularExponents;
    /** List of values for reflection indices */
    doubles reflectionIndices;
    /** List of values for opacities */
    doubles opacities;
    /** List of values for refraction indices */
    doubles refractionIndices;
    /** List of values for light emission */
    doubles emissions;
    /** List of values for glossiness */
    doubles glossinesses;
    /** List of values for casting user data */
    bools castUserData;
    /** List of values for shading modes */
    int32_ts shadingModes;
    /** List of values for user defined parameters */
    doubles userParameters;
    /** List of values for chameleon mode parameters */
    int32_ts chameleonModes;
} MaterialsDetails;

/**
 * @brief Structure containing information about how to build magnetic
 * fields from atom positions and charge
 *
 */
typedef struct
{
    /** Voxel size used to build the Octree acceleration structure */
    double voxelSize;
    /** Density of atoms to consider (Between 0 and 1) */
    double density;
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
    /** x, y, z coordinates stored in binary representation (4 byte double)
     */
    xyz_binary = 1,
    /** x, y, z coordinates and radius stored in binary representation (4
       byte double) */
    xyzr_binary = 2,
    /** x, y, z coordinates, radius, and charge stored in binary
       representation (4 byte double) */
    xyzrv_binary = 3,
    /** x, y, z coordinates stored in space separated ascii representation.
       One line per atom*/
    xyz_ascii = 4,
    /** x, y, z coordinates and radius stored in space separated ascii
       representation. One line per atom*/
    xyzr_ascii = 5,
    /** x, y, z coordinates, radius, and charge stored in space separated
       ascii representation. One line per atom*/
    xyzrv_ascii = 6
};

/**
 * @brief Structure defining how to export data into a file
 *
 */
typedef struct
{
    std::string filename;
    doubles lowBounds;
    doubles highBounds;
    XYZFileFormat fileFormat;
} FileAccessDetails;

/**
 * @brief Structure defining how to export data into a DB
 *
 */
typedef struct
{
    int32_t brickId;
    doubles lowBounds;
    doubles highBounds;
} DatabaseAccessDetails;

/**
 * @brief Structure defining how to build a point cloud from the scene
 *
 */
typedef struct
{
    double radius;
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
    Vector3d sceneSize;
    /** Number of bricks per side of the scene */
    uint32_t nbBricks;
    /** Size of the each brick in the scene */
    Vector3d brickSize;
} OOCSceneConfigurationDetails;

/**
 * @brief List of metrics for the current scene
 *
 */
typedef struct
{
    /** Number of models */
    uint32_t nbModels{0};
    /** Number of materials */
    uint32_t nbMaterials{0};
    /** Number of spheres */
    uint32_t nbSpheres{0};
    /** Number of cylinders */
    uint32_t nbCylinders{0};
    /** Number of cones */
    uint32_t nbCones{0};
    /** Number of triangle mesh vertices */
    uint32_t nbVertices{0};
    /** Number of triangle mesh indices */
    uint32_t nbIndices{0};
    /** Number of triangle mesh normals */
    uint32_t nbNormals{0};
    /** Number of triangle mesh colors */
    uint32_t nbColors{0};
} SceneInformationDetails;

/**
 * @brief Brain atlas
 *
 */
typedef struct
{
    /** Name of the assembly containing the atlas */
    std::string assemblyName;
    /** Load cells if set to true */
    bool loadCells{true};
    /** Cell radius **/
    double cellRadius{1.f};
    /** Load region meshes if set to true */
    bool loadMeshes{true};
    /** SQL filter for cells (WHERE condition) */
    std::string cellSqlFilter;
    /** SQL filter for regions (WHERE condition) */
    std::string regionSqlFilter;
    /** Scale of the atlas in the scene */
    doubles scale;
    /** Mesh transformation */
    doubles meshPosition;
    doubles meshRotation;
    doubles meshScale;
} AtlasDetails;

/**
 * @brief Color schemes that can be applied to vasculatures
 *
 */
enum class VasculatureColorScheme
{
    /** All edges use the same color */
    none = 0,
    /** Colored by node */
    node = 1,
    /** Colored by section */
    section = 2,
    /** Colored by sub-graph */
    subgraph = 3,
    /** Colored by pair */
    pair = 4,
    /** Colored by entry node */
    entry_node = 5,
    /** Colored by radius */
    radius = 6,
    /** Colored by point order within a section */
    section_points = 7,
    /** Colored by section orientation */
    section_orientation = 8
};

enum class VasculatureQuality
{
    low = 0,
    medium = 1,
    high = 2
};

typedef struct
{
    /** Name of the assembly containing the vasculature */
    std::string assemblyName;
    /** Population name */
    std::string populationName;
    /** Color scheme **/
    VasculatureColorScheme colorScheme;
    /** Use Signed Distance Fields as geometry */
    bool useSdf;
    /** Node gids to load. All if empty */
    uint32_ts gids;
    /** Geometry quality */
    VasculatureQuality quality;
    /** Multiplies the vasculature section radii by the specified value */
    double radiusMultiplier;
    /** SQL filter (WHERE condition) */
    std::string sqlFilter;
    /** Scale of the vasculature in the scene */
    doubles scale;
} VasculatureDetails;

typedef struct
{
    /** Name of the assembly containing the vasculature */
    std::string assemblyName;
    /** Name of the population on which the report applies */
    std::string populationName;
    /** Simulation report ID */
    uint64_t simulationReportId;
} VasculatureReportDetails;

typedef struct
{
    /** Name of the assembly containing the vasculature */
    std::string assemblyName;
    /** Name of the population on which the report applies */
    std::string populationName;
    /** Simulation report ID */
    uint64_t simulationReportId;
    /** Simulation frame number */
    uint64_t frame;
    /** Amplitude applied to the radius */
    double amplitude;
} VasculatureRadiusReportDetails;

enum class PopulationColorScheme
{
    /** All nodes use the same color */
    none = 0,
    /** Colored by id */
    id = 1
};

enum class MorphologyColorScheme
{
    /** All sections use the same color */
    none = 0,
    /** Colored by section */
    section = 1
};

enum class GeometryQuality
{
    low = 0,
    medium = 1,
    high = 2
};

typedef struct
{
    /** Name of the assembly containing the astrocytes */
    std::string assemblyName;
    /** Name of the population of astrocytes */
    std::string populationName;
    /** Name of the vasculature population. If not empty, endfeet are
     * automatically loaded */
    std::string vasculaturePopulationName;
    /** Load somas if set to true */
    bool loadSomas{true};
    /** Load dendrites if set to true */
    bool loadDendrites{true};
    /** Generate internal components (nucleus and mitochondria) */
    bool generateInternals{false};
    /** Use Signed Distance Fields as geometry */
    bool useSdf{false};
    /** Geometry quality */
    GeometryQuality geometryQuality;
    /** Geometry color scheme */
    MorphologyColorScheme morphologyColorScheme;
    /** Population color scheme */
    PopulationColorScheme populationColorScheme;
    /** Multiplies the astrocyte section radii by the specified value */
    double radiusMultiplier;
    /** SQL filter (WHERE condition) */
    std::string sqlFilter;
    /** Scale of the astrocyte in the scene */
    doubles scale;
    /** Extra optional parameters for astrocytes animation */
    doubles animationParams;
} AstrocytesDetails;

enum class NeuronSectionType
{
    undefined = 0,
    soma = 1,
    axon = 2,
    basal_dendrite = 3,
    apical_dendrite = 4
};

typedef struct
{
    /** Name of the assembly containing the astrocytes */
    std::string assemblyName;
    /** Name of the population of astrocytes */
    std::string populationName;
    /** Load somas if set to true */
    bool loadSomas{true};
    /** Load axons if set to true */
    bool loadAxon{true};
    /** Load bascal dendrites if set to true */
    bool loadBasalDendrites{true};
    /** Load apical dendrites if set to true */
    bool loadApicalDendrites{true};
    /** Load synapses if set to true */
    bool loadSynapses{false};
    /** Generate internal components (nucleus and mitochondria) */
    bool generateInternals{false};
    /** Generate external components (myelin steath) */
    bool generateExternals{false};
    /** Show membrane (Typically used to isolate internal and external
     * components*/
    bool showMembrane{true};
    /** Generates random varicosities along the axon */
    bool generateVaricosities{false};
    /** Use Signed Distance Fields as geometry */
    bool useSdf{false};
    /** Geometry quality */
    GeometryQuality geometryQuality;
    /** Geometry color scheme */
    MorphologyColorScheme morphologyColorScheme;
    /** Population color scheme */
    PopulationColorScheme populationColorScheme;
    /** Multiplies the astrocyte section radii by the specified value */
    double radiusMultiplier;
    /** SQL filter for nodes (WHERE condition) */
    std::string sqlNodeFilter;
    /** SQL filter dor sections (WHERE condition) */
    std::string sqlSectionFilter;
    /** Scale of the neuron in the scene */
    doubles scale;
    /** Extra optional parameters for neuron animation */
    doubles animationParams;
} NeuronsDetails;

typedef struct
{
    /** Name of the assembly containing the neurons */
    std::string assemblyName;
    /** Neuron identifier */
    uint64_t neuronId;
    /** Section identifier */
    uint64_t sectionId;
} NeuronIdSectionIdDetails;

typedef struct
{
    /** Name of the assembly containing the neurons */
    std::string assemblyName;
    /** Neuron identifier */
    uint64_t neuronId;
} NeuronIdDetails;

typedef struct
{
    bool status{true};
    doubles points;
} NeuronPointsDetails;

typedef struct
{
    doubles source;
    doubles target;
} LookAtDetails;

typedef struct
{
    doubles rotation;
} LookAtResponseDetails;

typedef struct
{
    /** Name of the assembly containing the white matter */
    std::string assemblyName;
    /** Name of the white matter population  */
    std::string populationName;
    /** Streamline radius */
    double radius{1.0};
    /** SQL filter for streamlines (WHERE condition) */
    std::string sqlFilter;
    /** Scale of the streamlines in the scene */
    doubles scale;
} WhiteMatterDetails;
} // namespace details

namespace common
{
class Node;
using NodePtr = std::shared_ptr<Node>;
using NodeMap = std::map<std::string, NodePtr>;

class Assembly;
using AssemblyPtr = std::shared_ptr<Assembly>;
using AssemblyMap = std::map<std::string, AssemblyPtr>;

enum class AssemblyConstraintType
{
    inside = 0,
    outside = 1
};
using AssemblyConstraint = std::pair<AssemblyConstraintType, AssemblyPtr>;
using AssemblyConstraints = std::vector<AssemblyConstraint>;

typedef struct
{
    Vector3d position;
    double radius;
    uint64_t sectionId{0};
    uint64_t graphId{0};
    uint64_t type{0};
    uint64_t pairId{0};
    uint64_t entryNodeId{0};
} GeometryNode;
using GeometryNodes = std::map<uint64_t, GeometryNode>;
using GeometryEdges = std::map<uint64_t, uint64_t>;
using Bifurcations = std::map<uint64_t, uint64_ts>;

// Thread safe container
class ThreadSafeContainer;
using ThreadSafeContainers = std::vector<ThreadSafeContainer>;

// SDF structures
class SDFGeometries;
using SDFGeometriesPtr = std::shared_ptr<SDFGeometries>;

struct SDFMorphologyData
{
    std::vector<SDFGeometry> geometries;
    std::vector<std::set<size_t>> neighbours;
    size_ts materials;
    size_ts localToGlobalIdx;
    size_ts bifurcationIndices;
    std::unordered_map<size_t, int> geometrySection;
    std::unordered_map<int, size_ts> sectionGeometries;
};
} // namespace common

namespace molecularsystems
{
using ModelDescriptors = std::vector<ModelDescriptorPtr>;

class Membrane;
using MembranePtr = std::shared_ptr<Membrane>;

class Protein;
using ProteinPtr = std::shared_ptr<Protein>;
using ProteinMap = std::map<std::string, ProteinPtr>;
using Proteins = std::vector<ProteinPtr>;

class Glycans;
using GlycansPtr = std::shared_ptr<Glycans>;
using GlycansMap = std::map<std::string, GlycansPtr>;

class RNASequence;
using RNASequencePtr = std::shared_ptr<RNASequence>;
using RNASequenceMap = std::map<std::string, RNASequencePtr>;

class EnzymeReaction;
using EnzymeReactionPtr = std::shared_ptr<EnzymeReaction>;
using EnzymeReactionMap = std::map<std::string, EnzymeReactionPtr>;

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
    Vector3d position;
    /** Occupancy */
    double occupancy;
    /** Temperature factor */
    double tempFactor;
    /** Element symbol */
    std::string element;
    /** Charge */
    std::string charge;
    /** Radius */
    double radius;
} Atom;
using AtomMap = std::multimap<size_t, Atom, std::less<size_t>>;

/**
 * @brief Sequence of residues
 *
 */
typedef struct
{
    /** Number of residues in the chain */
    size_t numRes;
    /** Residue name */
    strings resNames;
    /** Atom Offset */
    size_t offset;
} ResidueSequence;
using ResidueSequenceMap = std::map<std::string, ResidueSequence>;

/**
 * @brief Bonds map
 *
 */
using BondsMap = std::map<size_t, size_ts>;

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
using AminoAcidMap = std::map<std::string, AminoAcid>;

/**
 * @brief Set of residue names
 *
 */
using Residues = std::set<std::string>;

/**
 * @brief Atom radii in microns
 *
 */
using AtomicRadii = std::map<std::string, double>;

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
} // namespace molecularsystems

namespace atlas
{
class Atlas;
using AtlasPtr = std::shared_ptr<Atlas>;
} // namespace atlas

namespace vasculature
{
class Vasculature;
using VasculaturePtr = std::shared_ptr<Vasculature>;
} // namespace vasculature

namespace morphology
{
class Morphologies;
using MorphologiesPtr = std::shared_ptr<Morphologies>;
class Astrocytes;
using AstrocytesPtr = std::shared_ptr<Astrocytes>;
class Neurons;
using NeuronsPtr = std::shared_ptr<Neurons>;

typedef struct
{
    Vector3d center;
    double radius;
    uint64_ts children;
} AstrocyteSoma;
using AstrocyteSomaMap = std::map<uint64_t, AstrocyteSoma>;

typedef struct
{
    Vector3d position;
    Quaterniond rotation;
    uint64_t eType{0};
    uint64_t mType{0};
    uint64_t layer{0};
    uint64_t morphologyId{0};
} NeuronSoma;
using NeuronSomaMap = std::map<uint64_t, NeuronSoma>;

typedef struct
{
    uint64_t preSynapticNeuron;
    uint64_t postSynapticNeuron;
    Vector3d surfacePosition;
    Vector3d centerPosition;
} Synapse;
using SynapseMap = std::map<uint64_t, Synapse>;

typedef struct
{
    Vector4fs points;
    size_t type;
    int64_t parentId;
} Section;
using SectionMap = std::map<uint64_t, Section>;

typedef struct
{
    uint64_t vasculatureSectionId;
    uint64_t vasculatureSegmentId;
    double length;
    double radius;
    Vector4fs nodes;
} EndFoot;
using EndFootMap = std::map<uint64_t, EndFoot>;

typedef struct
{
    Vector3d position;
    Quaterniond rotation;
    uint64_t type{0};
    int64_t eType{0};
    uint64_t region{0};
} Cell;
using CellMap = std::map<uint64_t, Cell>;

} // namespace morphology

namespace connectomics
{
class WhiteMatter;
using WhiteMatterPtr = std::shared_ptr<WhiteMatter>;
using WhiteMatterStreamlines = std::vector<Vector3fs>;
} // namespace connectomics

namespace io
{
// Out of core brick manager
class OOCManager;
using OOCManagerPtr = std::shared_ptr<OOCManager>;

namespace db
{
class DBConnector;
using DBConnectorPtr = std::shared_ptr<DBConnector>;

typedef struct
{
    std::string description;
    double startTime;
    double endTime;
    double timeStep;
    std::string timeUnits;
    std::string dataUnits;
} SimulationReport;

namespace fields
{
class FieldsHandler;
typedef std::shared_ptr<FieldsHandler> FieldsHandlerPtr;
} // namespace fields

} // namespace db
} // namespace io
} // namespace bioexplorer
