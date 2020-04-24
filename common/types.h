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

#ifndef COVID19_TYPES_H
#define COVID19_TYPES_H

#include <brayns/common/mathTypes.h>
#include <brayns/common/types.h>
#include <brayns/engineapi/Scene.h>

#include <map>
#include <set>
#include <string>

// Model content types
enum class ModelContentType
{
    pdb = 0,
    obj = 1
};

// Color schemes
enum class ColorScheme
{
    none = 0,
    atoms = 1,
    chains = 2,
    residues = 3,
    amino_acid_sequence = 4,
    glycosylation_site = 5
};

struct ColorSchemeDescriptor
{
    std::string assemblyName;
    std::string name;
    ColorScheme colorScheme;
    std::vector<float> palette;
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
struct AssemblyDescriptor
{
    std::string name;
    std::vector<float> position;
    bool halfStructure;
};

class Assembly;
typedef std::shared_ptr<Assembly> AssemblyPtr;
typedef std::map<std::string, AssemblyPtr> AssemblyMap;

// Protein
struct ProteinDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string contents;
    float assemblyRadius;
    float atomRadiusMultiplier;
    bool loadBonds;
    bool addSticks;
    std::vector<size_t> chainIds;
    bool recenter;
    size_t occurrences;
    size_t randomSeed;
    float locationCutoffAngle;
    std::vector<float> orientation;
};

class Protein;
typedef std::shared_ptr<Protein> ProteinPtr;
typedef std::map<std::string, ProteinPtr> ProteinMap;

// Mesh
struct MeshDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string contents;
    float assemblyRadius;
    bool recenter;
    size_t occurrences;
    size_t randomSeed;
    std::vector<float> orientation;
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
    float assemblyRadius;
    float radius;
    std::vector<float> range;
    std::vector<float> params;
};

// Amino acid
struct AminoAcidSequenceDescriptor
{
    std::string assemblyName;
    std::string name;
    std::string aminoAcidSequence;
};

struct AminoAcidSequencesDescriptor
{
    std::string assemblyName;
    std::string name;
};

// Node
class Node;
typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

struct LoaderParameters
{
    // Radius multiplier
    float radiusMultiplier;
    // Color scheme to be applied to the proteins
    // [none|atoms|chains|residues|transmembrane_sequence|glycosylation_site]
    ColorScheme colorScheme;
    // Sequence of amino acids located in the virus membrane
    std::string aminoAcidSequence;
};

struct Atom
{
    std::string name;
    std::string altLoc;
    std::string resName;
    std::string chainId;
    size_t reqSeq;
    std::string iCode;
    brayns::Vector3f position;
    float occupancy;
    float tempFactor;
    std::string element;
    std::string charge;
    float radius;
};
typedef std::map<size_t, Atom> AtomMap;

// Amino acid sequence
struct Sequence
{
    size_t serNum;
    size_t numRes;
    std::vector<std::string> resNames;
};
typedef std::map<std::string, Sequence> SequenceMap;

// Bonds
typedef std::map<size_t, std::vector<size_t>> BondsMap;

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

typedef brayns::Vector3f Color;
typedef std::vector<Color> Palette;

#endif // COVID19_TYPES_H
