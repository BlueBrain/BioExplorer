#ifndef BIOEXPLORER_MOLECULE_H
#define BIOEXPLORER_MOLECULE_H

#include <plugin/bioexplorer/Node.h>
#include <plugin/common/Types.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
using namespace brayns;

const std::string KEY_UNDEFINED = "Undefined";
const std::string KEY_ATOM = "ATOM";
const std::string KEY_HETATM = "HETATM";
const std::string KEY_HEADER = "HEADER";
const std::string KEY_TITLE = "TITLE";
const std::string KEY_CONECT = "CONECT";
const std::string KEY_SEQRES = "SEQRES";
const std::string KEY_REMARK = "REMARK";

/**
 * @brief The Molecule class
 */
class Molecule : public Node
{
public:
    Molecule(Scene& scene, const size_ts& chainIds);

    StringMap getSequencesAsString() const;

protected:
    void _setAtomColorScheme();
    void _setChainColorScheme(const Palette& palette);
    void _setResiduesColorScheme(const Palette& palette);
    void _setAminoAcidSequenceColorScheme(const Palette& palette);
    void _setMaterialDiffuseColor(const size_t atomIndex,
                                  const RGBColor& color);
    void _setMaterialDiffuseColor(const size_t atomIndex, const Color& color);

    // Geometry
    void _buildModel(const std::string& assemblyName, const std::string& name,
                     const std::string& title, const std::string& header,
                     const ProteinRepresentation& representation,
                     const float atomRadiusMultiplier, const bool loadBonds);

    // IO
    void _readAtom(const std::string& line, const bool loadHydrogen);
    void _readSequence(const std::string& line);
    std::string _readHeader(const std::string& line);
    std::string _readTitle(const std::string& line);
    void _readRemark(const std::string& line);
    void _readConnect(const std::string& line);
    bool _loadChain(const size_t chainId);

    Scene& _scene;
    AtomMap _atomMap;
    Residues _residues;
    std::string _aminoAcidSequence;
    SequenceMap _sequenceMap;
    BondsMap _bondsMap;
    size_ts _chainIds;
    Vector2ui _aminoAcidRange;
    Boxf _bounds;
};
} // namespace bioexplorer
#endif // BIOEXPLORER_MOLECULE_H
