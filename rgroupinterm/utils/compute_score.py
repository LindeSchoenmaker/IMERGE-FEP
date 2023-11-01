import lomap
from rdkit import RDLogger
from rdkit.Chem import AllChem as _AllChem

RDLogger.DisableLog('rdApp.*')

def computeLOMAPScore(lig1, lig2, single_top = True):
    """Computes the LOMAP score for two input ligands, see https://github.com/OpenFreeEnergy/Lomap/blob/main/lomap/mcs.py."""
    _AllChem.EmbedMolecule(lig1, useRandomCoords=True)
    _AllChem.EmbedMolecule(lig2, useRandomCoords=True)

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