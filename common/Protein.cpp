#include "Protein.h"

#include <common/log.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

#include <fstream>

// Atomic radii in microns
static AtomicRadii atomicRadii = {{{"C"}, {67.f}},
                                  {{"N"}, {56.f}},
                                  {{"O"}, {48.f}},
                                  {{"H"}, {53.f}},
                                  {{"B"}, {87.f}},
                                  {{"F"}, {42.f}},
                                  {{"P"}, {98.f}},
                                  {{"S"}, {88.f}},
                                  {{"V"}, {171.f}},
                                  {{"K"}, {243.f}},
                                  {{"HE"}, {31.f}},
                                  {{"LI"}, {167.f}},
                                  {{"BE"}, {112.f}},
                                  {{"NE"}, {38.f}},
                                  {{"NA"}, {190.f}},
                                  {{"MG"}, {145.f}},
                                  {{"AL"}, {118.f}},
                                  {{"SI"}, {111.f}},
                                  {{"CL"}, {79.f}},
                                  {{"AR"}, {71.f}},
                                  {{"CA"}, {194.f}},
                                  {{"SC"}, {184.f}},
                                  {{"TI"}, {176.f}},
                                  {{"CR"}, {166.f}},
                                  {{"MN"}, {161.f}},
                                  {{"FE"}, {156.f}},
                                  {{"CO"}, {152.f}},
                                  {{"NI"}, {149.f}},
                                  {{"CU"}, {145.f}},
                                  {{"ZN"}, {142.f}},
                                  {{"GA"}, {136.f}},
                                  {{"GE"}, {125.f}},
                                  {{"AS"}, {114.f}},
                                  {{"SE"}, {103.f}},
                                  {{"BR"}, {94.f}},
                                  {{"KR"}, {88.f}},
                                  // TODO
                                  {{"OD1"}, {25.f}},
                                  {{"OD2"}, {25.f}},
                                  {{"CG1"}, {25.f}},
                                  {{"CG2"}, {25.f}},
                                  {{"CD1"}, {25.f}},
                                  {{"CB"}, {25.f}},
                                  {{"CG"}, {25.f}},
                                  {{"CD"}, {25.f}},
                                  {{"OE1"}, {25.f}},
                                  {{"NE2"}, {25.f}},
                                  {{"CZ"}, {25.f}},
                                  {{"NH1"}, {25.f}},
                                  {{"NH2"}, {25.f}},
                                  {{"CD2"}, {25.f}},
                                  {{"CE1"}, {25.f}},
                                  {{"CE2"}, {25.f}},
                                  {{"CE"}, {25.f}},
                                  {{"NZ"}, {25.f}},
                                  {{"OH"}, {25.f}},
                                  {{"CE"}, {25.f}},
                                  {{"ND1"}, {25.f}},
                                  {{"ND2"}, {25.f}},
                                  {{"OXT"}, {25.f}},
                                  {{"OG1"}, {25.f}},
                                  {{"NE1"}, {25.f}},
                                  {{"CE3"}, {25.f}},
                                  {{"CZ2"}, {25.f}},
                                  {{"CZ3"}, {25.f}},
                                  {{"CH2"}, {25.f}},
                                  {{"OE2"}, {25.f}},
                                  {{"OG"}, {25.f}},
                                  {{"OE2"}, {25.f}},
                                  {{"SD"}, {25.f}},
                                  {{"SG"}, {25.f}},
                                  {{"C1*"}, {25.f}},
                                  {{"C2"}, {25.f}},
                                  {{"C2*"}, {25.f}},
                                  {{"C3*"}, {25.f}},
                                  {{"C4"}, {25.f}},
                                  {{"C4*"}, {25.f}},
                                  {{"C5"}, {25.f}},
                                  {{"C5*"}, {25.f}},
                                  {{"C5M"}, {25.f}},
                                  {{"C6"}, {25.f}},
                                  {{"C8"}, {25.f}},
                                  {{"H1"}, {25.f}},
                                  {{"H1*"}, {25.f}},
                                  {{"H2"}, {25.f}},
                                  {{"H2*"}, {25.f}},
                                  {{"H3"}, {25.f}},
                                  {{"H3*"}, {25.f}},
                                  {{"H3P"}, {25.f}},
                                  {{"H4"}, {25.f}},
                                  {{"H4*"}, {25.f}},
                                  {{"H5"}, {25.f}},
                                  {{"H5*"}, {25.f}},
                                  {{"H5M"}, {25.f}},
                                  {{"H6"}, {25.f}},
                                  {{"H8"}, {25.f}},
                                  {{"N1"}, {25.f}},
                                  {{"N2"}, {25.f}},
                                  {{"N3"}, {25.f}},
                                  {{"N4"}, {25.f}},
                                  {{"N6"}, {25.f}},
                                  {{"N7"}, {25.f}},
                                  {{"N9"}, {25.f}},
                                  {{"O1P"}, {25.f}},
                                  {{"O2"}, {25.f}},
                                  {{"O2P"}, {25.f}},
                                  {{"O3*"}, {25.f}},
                                  {{"O3P"}, {25.f}},
                                  {{"O4"}, {25.f}},
                                  {{"O4*"}, {25.f}},
                                  {{"O5*"}, {25.f}},
                                  {{"O6"}, {25.f}},
                                  {{"OXT"}, {25.f}},
                                  {{"P"}, 25.f}};

// Amino acids
static AminoAcidMap aminoAcidMap = {{"ALA", {"Alanine", "A"}},
                                    {"ARG", {"Arginine", "R"}},
                                    {"ASN", {"Asparagine", "N"}},
                                    {"ASP", {"Aspartic acid", "D"}},
                                    {"CYS", {"Cysteine", "C"}},
                                    {"GLN", {"Glutamine", "Q"}},
                                    {"GLU", {"Glutamic acid", "E"}},
                                    {"GLY", {"Glycine", "G"}},
                                    {"HIS", {"Histidine", "H"}},
                                    {"ILE", {"Isoleucine", "I"}},
                                    {"LEU", {"Leucine", "L"}},
                                    {"LYS", {"Lysine", "K"}},
                                    {"MET", {"Methionine", "M"}},
                                    {"PHE", {"Phenylalanine", "F"}},
                                    {"SER", {"Proline Pro P Serine", "S"}},
                                    {"THR", {"Threonine", "T"}},
                                    {"TRP", {"Tryptophan", "W"}},
                                    {"TYR", {"Tyrosine", "Y"}},
                                    {"VAL", {"Valine", "V"}}};

// Protein color maps
static RGBColorMap colorMap = {
    {"H", {0xDF, 0xDF, 0xDF}},        {"He", {0xD9, 0xFF, 0xFF}},
    {"Li", {0xCC, 0x80, 0xFF}},       {"Be", {0xC2, 0xFF, 0x00}},
    {"B", {0xFF, 0xB5, 0xB5}},        {"C", {0x90, 0x90, 0x90}},
    {"N", {0x30, 0x50, 0xF8}},        {"O", {0xFF, 0x0D, 0x0D}},
    {"F", {0x9E, 0x05, 0x1}},         {"Ne", {0xB3, 0xE3, 0xF5}},
    {"Na", {0xAB, 0x5C, 0xF2}},       {"Mg", {0x8A, 0xFF, 0x00}},
    {"Al", {0xBF, 0xA6, 0xA6}},       {"Si", {0xF0, 0xC8, 0xA0}},
    {"P", {0xFF, 0x80, 0x00}},        {"S", {0xFF, 0xFF, 0x30}},
    {"Cl", {0x1F, 0xF0, 0x1F}},       {"Ar", {0x80, 0xD1, 0xE3}},
    {"K", {0x8F, 0x40, 0xD4}},        {"Ca", {0x3D, 0xFF, 0x00}},
    {"Sc", {0xE6, 0xE6, 0xE6}},       {"Ti", {0xBF, 0xC2, 0xC7}},
    {"V", {0xA6, 0xA6, 0xAB}},        {"Cr", {0x8A, 0x99, 0xC7}},
    {"Mn", {0x9C, 0x7A, 0xC7}},       {"Fe", {0xE0, 0x66, 0x33}},
    {"Co", {0xF0, 0x90, 0xA0}},       {"Ni", {0x50, 0xD0, 0x50}},
    {"Cu", {0xC8, 0x80, 0x33}},       {"Zn", {0x7D, 0x80, 0xB0}},
    {"Ga", {0xC2, 0x8F, 0x8F}},       {"Ge", {0x66, 0x8F, 0x8F}},
    {"As", {0xBD, 0x80, 0xE3}},       {"Se", {0xFF, 0xA1, 0x00}},
    {"Br", {0xA6, 0x29, 0x29}},       {"Kr", {0x5C, 0xB8, 0xD1}},
    {"Rb", {0x70, 0x2E, 0xB0}},       {"Sr", {0x00, 0xFF, 0x00}},
    {"Y", {0x94, 0xFF, 0xFF}},        {"Zr", {0x94, 0xE0, 0xE0}},
    {"Nb", {0x73, 0xC2, 0xC9}},       {"Mo", {0x54, 0xB5, 0xB5}},
    {"Tc", {0x3B, 0x9E, 0x9E}},       {"Ru", {0x24, 0x8F, 0x8F}},
    {"Rh", {0x0A, 0x7D, 0x8C}},       {"Pd", {0x69, 0x85, 0x00}},
    {"Ag", {0xC0, 0xC0, 0xC0}},       {"Cd", {0xFF, 0xD9, 0x8F}},
    {"In", {0xA6, 0x75, 0x73}},       {"Sn", {0x66, 0x80, 0x80}},
    {"Sb", {0x9E, 0x63, 0xB5}},       {"Te", {0xD4, 0x7A, 0x00}},
    {"I", {0x94, 0x00, 0x94}},        {"Xe", {0x42, 0x9E, 0xB0}},
    {"Cs", {0x57, 0x17, 0x8F}},       {"Ba", {0x00, 0xC9, 0x00}},
    {"La", {0x70, 0xD4, 0xFF}},       {"Ce", {0xFF, 0xFF, 0xC7}},
    {"Pr", {0xD9, 0xFF, 0xC7}},       {"Nd", {0xC7, 0xFF, 0xC7}},
    {"Pm", {0xA3, 0xFF, 0xC7}},       {"Sm", {0x8F, 0xFF, 0xC7}},
    {"Eu", {0x61, 0xFF, 0xC7}},       {"Gd", {0x45, 0xFF, 0xC7}},
    {"Tb", {0x30, 0xFF, 0xC7}},       {"Dy", {0x1F, 0xFF, 0xC7}},
    {"Ho", {0x00, 0xFF, 0x9C}},       {"Er", {0x00, 0xE6, 0x75}},
    {"Tm", {0x00, 0xD4, 0x52}},       {"Yb", {0x00, 0xBF, 0x38}},
    {"Lu", {0x00, 0xAB, 0x24}},       {"Hf", {0x4D, 0xC2, 0xFF}},
    {"Ta", {0x4D, 0xA6, 0xFF}},       {"W", {0x21, 0x94, 0xD6}},
    {"Re", {0x26, 0x7D, 0xAB}},       {"Os", {0x26, 0x66, 0x96}},
    {"Ir", {0x17, 0x54, 0x87}},       {"Pt", {0xD0, 0xD0, 0xE0}},
    {"Au", {0xFF, 0xD1, 0x23}},       {"Hg", {0xB8, 0xB8, 0xD0}},
    {"Tl", {0xA6, 0x54, 0x4D}},       {"Pb", {0x57, 0x59, 0x61}},
    {"Bi", {0x9E, 0x4F, 0xB5}},       {"Po", {0xAB, 0x5C, 0x00}},
    {"At", {0x75, 0x4F, 0x45}},       {"Rn", {0x42, 0x82, 0x96}},
    {"Fr", {0x42, 0x00, 0x66}},       {"Ra", {0x00, 0x7D, 0x00}},
    {"Ac", {0x70, 0xAB, 0xFA}},       {"Th", {0x00, 0xBA, 0xFF}},
    {"Pa", {0x00, 0xA1, 0xFF}},       {"U", {0x00, 0x8F, 0xFF}},
    {"Np", {0x00, 0x80, 0xFF}},       {"Pu", {0x00, 0x6B, 0xFF}},
    {"Am", {0x54, 0x5C, 0xF2}},       {"Cm", {0x78, 0x5C, 0xE3}},
    {"Bk", {0x8A, 0x4F, 0xE3}},       {"Cf", {0xA1, 0x36, 0xD4}},
    {"Es", {0xB3, 0x1F, 0xD4}},       {"Fm", {0xB3, 0x1F, 0xBA}},
    {"Md", {0xB3, 0x0D, 0xA6}},       {"No", {0xBD, 0x0D, 0x87}},
    {"Lr", {0xC7, 0x00, 0x66}},       {"Rf", {0xCC, 0x00, 0x59}},
    {"Db", {0xD1, 0x00, 0x4F}},       {"Sg", {0xD9, 0x00, 0x45}},
    {"Bh", {0xE0, 0x00, 0x38}},       {"Hs", {0xE6, 0x00, 0x2E}},
    {"Mt", {0xEB, 0x00, 0x26}},       {"none", {0xFF, 0xFF, 0xFF}},
    {"selection", {0xFF, 0x00, 0x00}}};

std::string& ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::ptr_fun<int, int>(std::isgraph)));
    return s;
}

std::string& rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::ptr_fun<int, int>(std::isgraph))
                .base(),
            s.end());
    return s;
}

std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

Protein::Protein(brayns::Scene& scene, const std::string& name,
                 const std::string& filename, const float radiusMultiplier)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open())
        throw std::runtime_error("Could not open " + filename);

    size_t lineIndex{0};

    while (file.good())
    {
        std::string line;
        std::getline(file, line);
        if (line.find("ATOM") == 0 /* || line.find("HETATM") == 0*/)
            _readAtom(line);
        else if (line.find("SEQRES") == 0)
            _readSequence(line);
        else if (line.find("TITLE") == 0)
            _readTitle(line);
    }
    file.close();

    auto model = scene.createModel();

    // Build 3d models according to atoms positions (re-centered to origin)
    brayns::Boxf bounds;

    for (const auto& atom : _atoms)
        bounds.merge(atom.position);
    const auto& center = bounds.getCenter();

    for (const auto& atom : _atoms)
    {
        auto material =
            model->createMaterial(atom.serial, std::to_string(atom.serial));
        material->setDiffuseColor({1.f, 1.f, 1.f});
        model->addSphere(atom.serial, {atom.position - center,
                                       radiusMultiplier * atom.radius});
    }

    // Metadata
    brayns::ModelMetadata metadata;
    metadata["Title"] = _title;

    const auto& size = bounds.getSize();
    metadata["Size"] = std::to_string(size.x) + ", " + std::to_string(size.y) +
                       ", " + std::to_string(size.z) + " angstroms";

    for (const auto& sequence : getSequencesAsString())
        metadata["Amino Acid Sequence " + sequence.first] = sequence.second;

    _modelDescriptor =
        std::make_shared<brayns::ModelDescriptor>(std::move(model), name,
                                                  filename, metadata);
    // Transformation
    brayns::Transformation transformation;
    transformation.setRotationCenter(center);

    _modelDescriptor->setTransformation(transformation);

#if 0
    // Add cylinder
    const auto size = bounds.getSize();
    PLUGIN_ERROR << "size=" << size << std::endl;
    model->addCylinder(0, {{0, 0, 0}, {0, 0, size.z}, size.x});
#endif
}

StringMap Protein::getSequencesAsString() const
{
    StringMap sequencesAsStrings;
    for (const auto& sequence : _sequenceMap)
    {
        std::string shortSequence;
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        sequencesAsStrings[std::to_string(sequence.first)] = shortSequence;
        PLUGIN_INFO << sequence.first << " (" << sequence.second.resNames.size()
                    << "): " << shortSequence << std::endl;
    }
    return sequencesAsStrings;
}

void Protein::setColorScheme(const ColorScheme& colorScheme,
                             const Palette& palette)
{
    switch (colorScheme)
    {
    case ColorScheme::none:
        for (auto& atom : _atoms)
            _setMaterialDiffuseColor(atom.serial, colorMap[0]);
        break;
    case ColorScheme::atoms:
        _setAtomColorScheme();
        break;
    case ColorScheme::chains:
        _setChainColorScheme(palette);
        break;
    case ColorScheme::residues:
        _setResiduesColorScheme(palette);
        break;
    case ColorScheme::amino_acid_sequence:
        _setAminoAcidSequenceColorScheme(palette);
        break;
    case ColorScheme::glycosylation_site:
        _setGlycosylationSiteColorScheme(palette);
        break;
    default:
        PLUGIN_THROW(std::runtime_error("Unknown colorscheme"))
    }
}

void Protein::_setAtomColorScheme()
{
    std::set<size_t> materialId;
    for (const auto& atom : _atoms)
    {
        const size_t index = static_cast<size_t>(
            std::distance(colorMap.begin(), colorMap.find(atom.element)));
        materialId.insert(index);
        _setMaterialDiffuseColor(atom.serial, colorMap[atom.element]);
    }
    PLUGIN_INFO << "Applying atom color scheme (" << materialId.size() << ")"
                << std::endl;
}

void Protein::_setAminoAcidSequenceColorScheme(const Palette& palette)
{
    size_t atomCount = 0;
    for (const auto& sequence : _sequenceMap)
    {
        std::string shortSequence;
        for (const auto& resName : sequence.second.resNames)
            shortSequence += aminoAcidMap[resName].shortName;

        const auto sequencePosition = shortSequence.find(_aminoAcidSequence);
        if (sequencePosition != -1)
        {
            PLUGIN_INFO << _aminoAcidSequence << " was found at position "
                        << sequencePosition << std::endl;
            size_t minSeq = 1e6;
            size_t maxSeq = 0;
            for (auto& atom : _atoms)
            {
                minSeq = std::min(minSeq, atom.reqSeq);
                maxSeq = std::max(maxSeq, atom.reqSeq);
                if (atom.reqSeq >= sequencePosition &&
                    atom.reqSeq <
                        sequencePosition + _aminoAcidSequence.length())
                {
                    _setMaterialDiffuseColor(atom.serial, palette[1]);
                    ++atomCount;
                }
                else
                    _setMaterialDiffuseColor(atom.serial, palette[0]);
            }
            PLUGIN_DEBUG << atomCount << "[" << minSeq << "," << maxSeq
                         << "] atoms where colored" << std::endl;
        }
        else
            PLUGIN_WARN << _aminoAcidSequence << " was not found in "
                        << shortSequence << std::endl;
    }
    PLUGIN_INFO << "Applying Amino Acid Sequence color scheme ("
                << (atomCount > 0 ? "2" : "1") << ")" << std::endl;
}

void Protein::_setGlycosylationSiteColorScheme(const Palette& palette)
{
    bool found{false};
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        auto& atom = _atoms[i];
        _setMaterialDiffuseColor(atom.serial, palette[0]);

        if (atom.resName == "ASN")
            if (i + 2 < _atoms.size() && (_atoms[i + 2].resName == "THR" ||
                                          _atoms[i + 2].resName == "SER"))
            {
                found = true;
                _setMaterialDiffuseColor(atom.serial, palette[1]);
            }
    }
    PLUGIN_INFO << "Applying Glycosylation Site color scheme ("
                << (found ? "2" : "1") << ")" << std::endl;
}

void Protein::_setChainColorScheme(const Palette& palette)
{
    std::set<size_t> materialId;
    for (auto& atom : _atoms)
    {
        const size_t index = static_cast<size_t>(atom.chainId[0]) - 64;
        materialId.insert(index);
        _setMaterialDiffuseColor(atom.serial, palette[index]);
    }
    PLUGIN_INFO << "Applying Chain color scheme (" << materialId.size() << ")"
                << std::endl;
}

void Protein::_setResiduesColorScheme(const Palette& palette)
{
    std::set<size_t> materialId;
    for (auto& atom : _atoms)
    {
        const size_t index = static_cast<size_t>(
            std::distance(_residues.begin(), _residues.find(atom.resName)));
        materialId.insert(index);
        _setMaterialDiffuseColor(atom.serial, palette[index]);
    }
    PLUGIN_INFO << "Applying Residues color scheme (" << materialId.size()
                << ")" << std::endl;
}

void Protein::_setMaterialDiffuseColor(const size_t atomIndex,
                                       const RGBColor& color)
{
    auto& model = _modelDescriptor->getModel();
    auto material = model.getMaterial(atomIndex);
    if (material)
    {
        material->setDiffuseColor(
            {color.R / 255.f, color.G / 255.f, color.B / 255.f});
        material->commit();
    }
}

void Protein::_setMaterialDiffuseColor(const size_t atomIndex,
                                       const Color& color)
{
    auto& model = _modelDescriptor->getModel();
    auto material = model.getMaterial(atomIndex);
    if (material)
    {
        material->setDiffuseColor(color);
        material->commit();
    }
}

void Protein::_readAtom(const std::string& line)
{
    Atom atom;
    // --------------------------------------------------------------------
    // COLUMNS DATA TYPE    FIELD     DEFINITION
    // --------------------------------------------------------------------
    // 1 - 6   Record name  "ATOM "
    // 7 - 11  Integer      serial     Atom serial number
    // 13 - 16 Atom         name       Atom name
    // 17      Character    altLoc     Alternate location indicator
    // 18 - 20 Residue name resName    Residue name
    // 22      Character    chainID    Chain identifier
    // 23 - 26 Integer      resSeq     Residue sequence number
    // 27      AChar        iCode      Code for insertion of residues
    // 31 - 38 Real(8.3)    x          Orthogonal coords for X in angstroms
    // 39 - 46 Real(8.3)    y          Orthogonal coords for Y in Angstroms
    // 47 - 54 Real(8.3)    z          Orthogonal coords for Z in Angstroms
    // 55 - 60 Real(6.2)    occupancy  Occupancy
    // 61 - 66 Real(6.2)    tempFactor Temperature factor
    // 77 - 78 LString(2)   element    Element symbol, right-justified
    // 79 - 80 LString(2)   charge     Charge on the atom
    // --------------------------------------------------------------------

    atom.serial = static_cast<size_t>(atoi(line.substr(6, 4).c_str()));
    std::string s = line.substr(12, 4);
    atom.name = trim(s);

    s = line.substr(16, 1);
    atom.altLoc = trim(s);

    s = line.substr(17, 3);
    atom.resName = trim(s);

    _residues.insert(atom.resName);

    s = line.substr(21, 1);
    atom.chainId = trim(s);

    atom.reqSeq = static_cast<size_t>(atoi(line.substr(22, 4).c_str()));

    atom.iCode = line.substr(26, 1);

    atom.position.x = static_cast<float>(atof(line.substr(30, 8).c_str()));
    atom.position.y = static_cast<float>(atof(line.substr(38, 8).c_str()));
    atom.position.z = static_cast<float>(atof(line.substr(46, 8).c_str()));

    atom.occupancy = static_cast<float>(atof(line.substr(54, 6).c_str()));

    atom.tempFactor = static_cast<float>(atof(line.substr(60, 6).c_str()));

    s = line.substr(76, 2);
    atom.element = trim(s);

    s = line.substr(78, 2);
    atom.charge = trim(s);

    // Convert position from nanometers
    atom.position = 0.01f * atom.position;

    // Convert radius from angstrom
    atom.radius = 0.0001f * atomicRadii[atom.element];

    _atoms.push_back(atom);
}

void Protein::_readSequence(const std::string& line)
{
    // -------------------------------------------------------------------------
    // COLUMNS TYPE      FIELD    DEFINITION
    // -------------------------------------------------------------------------
    // 1 - 6   Record name "SEQRES"
    // 8 - 10  Integer   serNum   Serial number of the SEQRES record for the
    //                            current chain. Starts at 1 and increments by
    //                            one each line. Reset to 1 for each chain.
    // 12      Character chainID  Chain identifier. This may be any single legal
    //                            character, including a blank which is is used
    //                            if there is only one chain
    // 14 - 17 Integer   numRes   Number of residues in the chain. This value is
    //                            repeated on every record.
    // 20 - 22 String    resName  Residue name
    // 24 - 26 ...
    // -------------------------------------------------------------------------

    std::string s = line.substr(11, 1);
    size_t chainId = static_cast<size_t>(s[0]) - 64;

    Sequence& sequence = _sequenceMap[chainId];
    sequence.serNum = static_cast<size_t>(atoi(line.substr(7, 3).c_str()));
    sequence.numRes = static_cast<size_t>(atoi(line.substr(13, 4).c_str()));

    for (size_t i = 19; i < line.length(); i += 4)
    {
        s = line.substr(i, 4);
        s = trim(s);
        if (!s.empty())
            sequence.resNames.push_back(s);
    }
}

void Protein::_readTitle(const std::string& line)
{
    std::string s = line.substr(11);
    _title = trim(s);
}
