/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_TYPES_H
#define BIOEXPLORER_TYPES_H

#include <Defines.h>
#include <brayns/engineapi/Scene.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bioexplorer
{
using namespace brayns;

const std::string PLUGIN_VERSION = "0.6.0";

// Metadata
const std::string METADATA_ASSEMBLY = "Assembly";
const std::string METADATA_TITLE = "Title";
const std::string METADATA_HEADER = "Header";
const std::string METADATA_ATOMS = "Atoms";
const std::string METADATA_BONDS = "Bonds";
const std::string METADATA_SIZE = "Size";

// Model position randomization types
enum class PositionRandomizationType
{
    circular = 0,
    radial = 1
};

// Shapes
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

// Assembly
enum class AssemblyType
{
    protein = 0,
    membrane = 1,
    sugar = 3,
    rnaSequence = 4
};

struct AssemblyDescriptor
{
    std::string name;
    std::vector<float> position;
    std::vector<float> orientation;
    std::vector<float> clippingPlanes;
};

class Assembly;
typedef std::shared_ptr<Assembly> AssemblyPtr;
typedef std::map<std::string, AssemblyPtr> AssemblyMap;

struct AssemblyTransformationsDescriptor
{
    std::string assemblyName;
    std::string name;
    std::vector<float> transformations;
};

// Node
class Node;
typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

enum class ProteinRepresentation
{
    atoms = 0,
    atoms_and_sticks = 1,
    contour = 2,
    surface = 3
};

enum class AssemblyShape
{
    spherical = 0,
    planar = 1,
    sinusoidal = 2,
    cubic = 3,
    fan = 4,
    bezier = 5
};

// membrane
struct MembraneDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string content1;
    std::string content2;
    std::string content3;
    std::string content4;
    AssemblyShape shape;
    floats assemblyParams;
    float atomRadiusMultiplier;
    bool loadBonds;
    bool loadNonPolymerChemicals;
    ProteinRepresentation representation;
    size_ts chainIds;
    bool recenter;
    size_t occurrences;
    size_t randomSeed;
    float locationCutoffAngle;
    PositionRandomizationType positionRandomizationType;
    floats orientation;
};
class Membrane;
typedef std::shared_ptr<Membrane> MembranePtr;

// Protein
struct ProteinDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string contents;
    AssemblyShape shape;
    floats assemblyParams;
    float atomRadiusMultiplier;
    bool loadBonds;
    bool loadNonPolymerChemicals;
    bool loadHydrogen;
    ProteinRepresentation representation;
    size_ts chainIds;
    bool recenter;
    size_t occurrences;
    size_ts allowedOccurrences;
    size_t randomSeed;
    float locationCutoffAngle;
    PositionRandomizationType positionRandomizationType;
    floats position;
    floats orientation;
};
class Protein;
typedef std::shared_ptr<Protein> ProteinPtr;
typedef std::map<std::string, ProteinPtr> ProteinMap;

struct SugarsDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string contents;
    std::string proteinName;
    float atomRadiusMultiplier;
    bool addSticks;
    bool recenter;
    size_ts chainIds;
    size_ts siteIndices;
    size_ts allowedOccurrences;
    floats orientation;
};

class Glycans;
typedef std::shared_ptr<Glycans> GlycansPtr;
typedef std::map<std::string, GlycansPtr> GlycansMap;

// Mesh
struct MeshDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string contents;
    AssemblyShape shape;
    floats assemblyParams;
    bool recenter;
    size_t occurrences;
    size_t randomSeed;
    float locationCutoffAngle;
    PositionRandomizationType positionRandomizationType;
    floats position;
    floats orientation;
};
class Mesh;
typedef std::shared_ptr<Mesh> MeshPtr;
typedef std::map<std::string, MeshPtr> MeshMap;

// RNA sequence
struct RNASequenceDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string contents;
    RNAShape shape;
    floats assemblyParams;
    std::vector<float> range;
    std::vector<float> params;
};
class RNASequence;
typedef std::shared_ptr<RNASequence> RNASequencePtr;

// Amino acid
struct AminoAcidSequenceAsStringDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string sequence;
};

struct AminoAcidSequenceAsRangeDescriptor
{
    std::string assemblyName;
    std::string name;
    size_ts range;
};

struct AminoAcidInformationDescriptor
{
    std::string assemblyName;
    std::string name;
};

struct Atom
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
};
typedef std::multimap<size_t, Atom, std::less<size_t>> AtomMap;

// Amino acid sequence
struct Sequence
{
    //    size_t serNum;
    size_t numRes;
    std::vector<std::string> resNames;
};
typedef std::map<std::string, Sequence> SequenceMap;

// Bonds
typedef std::map<size_t, size_ts> BondsMap;

struct AminoAcid
{
    std::string name;
    std::string shortName;
    //    std::string sideChainClass;
    //    std::string sideChainPolarity;
    //    std::string sideChainCharge;
    //    size_t hidropathyIndex;
    //    size_t absorbance;
    //    float molarAttenuationCoefficient;
    //    float molecularMass;
    //    float occurenceInProtein;
    //    std::string standardGeneticCode;
};
typedef std::map<std::string, AminoAcid> AminoAcidMap;

// Residues
typedef std::set<std::string> Residues;

// Typedefs
typedef std::map<std::string, std::string> StringMap;

// Structure defining an atom radius in microns
typedef std::map<std::string, float> AtomicRadii;

// Structure defining the color of atoms according to the JMol Scheme
struct RGBColor
{
    short r, g, b;
};
typedef std::map<std::string, RGBColor> RGBColorMap;

typedef Vector3f Color;
typedef std::vector<Color> Palette;
typedef std::vector<Quaterniond> Quaternions;
typedef std::vector<Vector3f> Vector3fs;
typedef std::vector<std::pair<Vector3f, float>> OccupiedDirections;

const float BOND_RADIUS = 0.025f;
const float DEFAULT_STICK_DISTANCE = 0.16f;

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
static RGBColorMap atomColorMap = {
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

struct AddGrid
{
    float minValue;
    float maxValue;
    float steps;
    float radius;
    float planeOpacity;
    bool showAxis;
    bool useColors;
    std::vector<float> position;
};

// Color schemes and materials
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

struct ColorSchemeDescriptor
{
    std::string assemblyName;
    std::string name;
    ColorScheme colorScheme;
    std::vector<float> palette;
    size_ts chainIds;
};

struct MaterialIds
{
    std::vector<size_t> ids;
};

struct ModelId
{
    size_t modelId;
};

struct MaterialsDescriptor
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
};

// Movies and frames
struct CameraDefinition
{
    std::vector<double> origin;
    std::vector<double> direction;
    std::vector<double> up;
    double apertureRadius;
    double focusDistance;
    double interpupillaryDistance;
};

struct ExportFramesToDisk
{
    std::string path;
    std::string format;
    uint16_t quality{100};
    uint16_t spp{0};
    uint16_t startFrame{0};
    std::vector<uint64_t> animationInformation;
    std::vector<double> cameraInformation;
};

struct FrameExportProgress
{
    float progress;
};

// Fields
struct BuildFields
{
    float voxelSize;
};

// IO
struct ModelIdFileAccess
{
    size_t modelId;
    std::string filename;
};

struct FileAccess
{
    std::string filename;
};

} // namespace bioexplorer

#endif // BIOEXPLORER_TYPES_H
