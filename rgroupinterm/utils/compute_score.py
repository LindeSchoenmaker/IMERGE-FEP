
import lomap
from openeye import oechem, oeomega, oeshape
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')

def computeLOMAPScore(lig1, lig2, single_top = True):
    """Computes the LOMAP score for two input ligands, see https://github.com/OpenFreeEnergy/Lomap/blob/main/lomap/mcs.py."""
    AllChem.EmbedMolecule(lig1, useRandomCoords=True)
    AllChem.EmbedMolecule(lig2, useRandomCoords=True)

    MC = lomap.MCS(lig1, lig2, verbose=None)

    # # Rules calculations
    mcsr = MC.mcsr()
    strict = MC.tmcsr(strict_flag=True)
    loose = MC.tmcsr(strict_flag=False)
    mncar = MC.mncar()
    atnum = MC.atomic_number_rule()
    hybrid = MC.hybridization_rule()
    sulf = MC.sulfonamides_rule()
    if single_top:
        het = MC.heterocycles_rule()
        growring = MC.transmuting_methyl_into_ring_rule()
        changering = MC.transmuting_ring_sizes_rule()


    score = mncar * mcsr * atnum * hybrid
    score *= sulf
    if single_top:
        score *= het * growring
        lomap_score = score*changering
    else:
        lomap_score = score

    return lomap_score


def computeTanimotoScore(lig1, lig2):
    """
    Calculate the Tanimoto similarity for a list of molecules to a start and target molecule.
    Based on https://github.com/daanjiskoot/Intermediate_generator

    Args:
        start: starting molecule.
        intermediate: intermediate molecules.
        target: target molecule.

    Returns:
        list: List of similarity scores.
    """
    # calculate the ECFP4 fingerprints for the three molecules
    fp_1 = AllChem.GetMorganFingerprint(lig1, radius=2)
    fp_2 = AllChem.GetMorganFingerprint(lig2, radius=3)

    # calculate similarities
    similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)

    return similarity


def normalize_scores(scores):
    """Normalize a list of scores such that the maximum score maps to 0.5."""

    max_score = max(scores)

    # If the max score is 0, return the original scores (or handle accordingly)
    if max_score == 0:
        return [0] * len(scores)

    normalized_scores = [score / (2 * max_score) for score in scores]
    return normalized_scores


def rdmol_to_oemol(mol):
    # Create OE molecule
    smiles = Chem.MolToSmiles(mol)

    mol = oechem.OEMol()
    if not oechem.OEParseSmiles(mol, smiles):
        print("Couldn't parse smiles: %s" % smiles)
        return None
    return mol


def generate_best_conformer(rdmol):
    # Create OE molecule
    mol = rdmol_to_oemol(rdmol)

    # Generate conformers
    omega = initialize_omega()

    if not omega(mol):
        print("Omega failed")
        return None

    # Select the best conformer (i.e., the first one, as they're sorted by energy)
    mol = oechem.OEMol(mol.GetConf(oechem.OEHasConfIdx(0)))

    return mol


def initialize_omega():
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(100)
    omega.SetStrictStereo(False)
    omega.SetStrictAtomTypes(False)
    return omega


def getIntermediateMetrics(rdmol, noconfs, refmol):
    tancombolist = []
    shapetan = []
    colortan = []

    fitmol = rdmol_to_oemol(rdmol)
    options = oeshape.OEROCSOptions()
    options.SetNumBestHits(noconfs)
    options.SetConfsPerHit(noconfs)
    rocs = oeshape.OEROCS(options)
    omega = initialize_omega()
    omega(fitmol)
    fitmol.SetTitle('Analog')
    rocs.AddMolecule(fitmol)

    for res in rocs.Overlay(refmol):
        outmol = res.GetOverlayConfs()
        oeshape.OERemoveColorAtoms(outmol)
        oechem.OEAddExplicitHydrogens(outmol)
        tancombolist.append(res.GetTanimotoCombo())
        shapetan.append(res.GetShapeTanimoto())
        colortan.append(res.GetColorTanimoto())

    return tancombolist, shapetan, colortan


def computeROCSScore(lig1, lig2):
    # Preparation
    prep = oeshape.OEOverlapPrep()

    start_mol = generate_best_conformer(lig1)
    if start_mol is None or start_mol.NumAtoms() == 0:
        print("Problem with starting mol")
        return None
    # prepare molecule for overlap (optimize, add hydrogens, etc.)
    prep.Prep(start_mol)

    # Process the intermediate mol
    tancombolist_start, _, _ = getIntermediateMetrics(lig2, 1, start_mol)

    if not tancombolist_start:
        print("Error processing intermediate SMILES for ROCS.")
        return None

    score = tancombolist_start[0] / 2  # normalization

    return score
