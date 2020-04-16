#ifndef COVID19_TYPES_H
#define COVID19_TYPES_H

#include <brayns/common/mathTypes.h>
#include <brayns/common/types.h>

#include <map>
#include <set>
#include <string>

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

/** Structure defining an atom radius in microns
 */
typedef std::map<std::string, float> AtomicRadii;

/** Structure defining the color of atoms according to the JMol Scheme
 */
struct RGBColor
{
    short R, G, B;
};
typedef std::map<std::string, RGBColor> RGBColorMap;

typedef brayns::Vector3f Color;
typedef std::vector<Color> Palette;

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
    size_t serial;
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
typedef std::vector<Atom> Atoms;

struct Sequence
{
    size_t serNum;
    size_t numRes;
    std::vector<std::string> resNames;
};
typedef std::map<std::string, Sequence> SequenceMap;

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

// Protein
class Protein;
typedef std::shared_ptr<Protein> ProteinPtr;
typedef std::map<std::string, ProteinPtr> ProteinMap;

// Typedefs
typedef std::map<std::string, std::string> StringMap;

#endif // COVID19_TYPES_H
