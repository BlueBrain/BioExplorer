#ifndef COVID19_PROTEINLOADER_H
#define COVID19_PROTEINLOADER_H

#include <brayns/common/types.h>

#include <set>

enum class ColorScheme
{
    none = 0,
    atoms = 1,
    chains = 2,
    residues = 3,
    transmembrane_sequence = 4,
    glycosylation_site = 5
};

struct LoaderParameters
{
    // Radius multiplier
    float radiusMultiplier;
    // Color scheme to be applied to the proteins
    // [none|atoms|chains|residues|transmembrane_sequence|glycosylation_site]
    ColorScheme colorScheme;
    // Sequence of amino acids located in the virus membrane
    std::string transmembraneSequence;
};

struct Atom
{
    size_t index;
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
    size_t materialId;
};

struct Sequence
{
    size_t serNum;
    size_t numRes;
    std::vector<std::string> resNames;
};

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

// Typedefs
typedef std::vector<Atom> Atoms;
typedef std::set<std::string> Residues;
typedef std::map<size_t, Sequence> SequenceMap;
typedef std::map<std::string, AminoAcid> AminoAcidMap;

class ProteinLoader
{
public:
    ProteinLoader(brayns::Scene& scene);

    std::vector<std::string> getSupportedExtensions() const;
    brayns::PropertyMap getProperties() const;

    bool isSupported(const std::string& filename,
                     const std::string& extension) const;
    brayns::ModelDescriptorPtr importFromFile(
        const std::string& fileName, const LoaderParameters& loaderParameters);

private:
    void readAtom(const std::string& line, const ColorScheme colorScheme,
                  const float radiusMultiplier, Atoms& atoms,
                  Residues& residues) const;

    void readSequence(const std::string& line, SequenceMap& sequenceMap) const;

    brayns::Scene& _scene;
};
#endif // COVID19_PROTEINLOADER_H
