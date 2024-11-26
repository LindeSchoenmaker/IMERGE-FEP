import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign, rdMolDescriptors
from rdkit.Chem import rdRGroupDecomposition as rdRGD

p = AllChem.ETKDGv2()
p.randomSeed = 42
p.verbose = False

def get_3d_core(file):
    suppl = Chem.SDMolSupplier(file)

    core = rdFMCS.FindMCS(suppl)
    qry = Chem.MolFromSmarts(core.smartsString)
    result_dict = rdRGD.RGroupDecompose([qry], [suppl[0]])
    return result_dict[0][0]['Core']


def set_coords_highest_overlap(mol, reference_set):
    overlap = 0
    mol = Chem.AddHs(mol)
    for reference in reference_set:
        core = rdFMCS.FindMCS(
            [mol, reference],
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
            ringCompare=Chem.rdFMCS.RingCompare.PermissiveRingFusion,
            timeout=2)
        qry = Chem.MolFromSmarts(core.smartsString)
        
        if qry.GetNumHeavyAtoms() > overlap:
            overlap = qry.GetNumHeavyAtoms()
            result_dict = rdRGD.RGroupDecompose([qry], [reference])
            core_3d = result_dict[0][0]['Core']
            core_3d = Chem.DeleteSubstructs(core_3d, Chem.MolFromSmarts('[#0]'))
            mol = AllChem.ConstrainedEmbed(mol, core_3d)
    return mol

def optimize_conformation(mol, addHs=True):
    """optimize conformation of molecule 
    mol: is rdkit molecule object
    addHs: whether to add hydrogens or not
    """
    if addHs:
        mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol,randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    return mol

def get_mcs(mylist):
    """Get MCS of molecules in list, input should be a list"""
    # calc MCS
    res=rdFMCS.FindMCS(mylist,threshold=0.9, matchValences=True, ringMatchesRingOnly=True, completeRingsOnly=True)
    p = Chem.MolFromSmarts(res.smartsString)
    # check if all ligands match the mcs
    matchingMols = [x for x in mylist if x.HasSubstructMatch(p)]
    if len(matchingMols) != len(mylist):
        print('Warning: not all molecules contain the MCS')
    core = AllChem.DeleteSubstructs(AllChem.ReplaceSidechains(Chem.RemoveHs(matchingMols[1]),p),Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()

    return core

def align_conformers(ref, mol, core, addHs=True):
    """Align and score conformers of mol to reference, returns best pair
    ref: is rdkit molecule of template / reference
    mol: is rdkit molecule object
    core: is MCS"""
    # get indices of reference molecule’s atoms that match core
    ref_match = ref.GetSubstructMatch(core)

    if addHs:
        mol = Chem.AddHs(mol)

    # get mapping of attom id & it's 3d coords
    mol_match = mol.GetSubstructMatch(core)
    cmap = {mol_match[i]:ref.GetConformer().GetAtomPosition(ref_match[i]) for i in range(len(ref_match))}

    #create multiple conformers with coordmap as restriction
    AllChem.EmbedMultipleConfs(mol, 500, randomSeed=42, coordMap=cmap)

    # align molecule to reference and keep best alignment
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    crippen_ref = rdMolDescriptors._CalcCrippenContribs(ref)

    tempscore = []
    ids = []
    for prb_cid in range(mol.GetNumConformers()):
        crippenO3A = rdMolAlign.GetCrippenO3A(prbMol = mol, refMol = ref, prbCrippenContribs = crippen_contribs,
                                            refCrippenContribs = crippen_ref, prbCid = prb_cid, refCid = 0)
        crippenO3A.Align()
        tempscore.append(crippenO3A.Score())
        ids.append([prb_cid])
    best = np.argmax(tempscore)
    # select best isomer
    best_mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol, confId=int(ids[best][0])), removeHs=False)

    return ref, best_mol


def just_optimized_conformation(ref, to_align, intermediate):
    """Code for optimizing the conformation, generating conformers and aligning conformers to reference."""

    # get lowest energy conformation of template
    ref = optimize_conformation(ref, addHs=True)

    core = Chem.MolFromSmarts('[A]')

    _, to_align = align_conformers(ref, to_align, core)

    ref, intermediate_new = align_conformers(ref, intermediate, core)

    return ref, to_align, intermediate_new


def main(ref, to_align, intermediates, reference_set=None):
    """Code for optimizing the conformation, generating conformers and aligning conformers to reference."""

    if reference_set is None:
        # get lowest energy conformation of template
        ref = optimize_conformation(ref, addHs=True)

        # get MCS ref and to align
        core = get_mcs([to_align, ref])
        # align to reference
        _, to_align = align_conformers(ref, to_align, core)

        new_itermediates = []
        for intermediate in intermediates:
            # get MCS core
            core = get_mcs([intermediate, ref])
            # align to reference
            ref, intermediate_new = align_conformers(ref, intermediate, core)
            new_itermediates.append(intermediate_new)

        return ref, to_align, new_itermediates
    else:
        # get MCS ref & core
        ref = set_coords_highest_overlap(ref, reference_set)
        to_align = set_coords_highest_overlap(to_align, reference_set)
        new_intermediates = []
        for intermediate in intermediates:
            intermediate = set_coords_highest_overlap(intermediate, reference_set)
            new_intermediates.append(intermediate)
        return ref, to_align, new_intermediates


if __name__ == "__main__":
    with open("rbfe_gromacs/input/rhfe_pairs.json") as file:
        lig_dict = json.load(file)

    suppl = Chem.SDMolSupplier('rbfe_gromacs/input/ligands_cdk8.sdf')

    for i in range(1, 8):
        # load molecules
        sdf = True
        smiles_parents = [lig_dict[x] for x in lig_dict.keys() if f"{i}P" in x]
        ligands  = [Chem.MolFromSmiles(x) for x in smiles_parents]
        smiles_intermediates = [lig_dict[x] for x in lig_dict.keys() if f"{i}I" in x]
        intermediates = [Chem.MolFromSmiles(x) for x in smiles_intermediates]

        # first entry in list of ligands is reference, second ligand from pair will be aligned to this
        ref = ligands[0]
        to_align = ligands[1]
        ref, to_align, intermediates = main(ref, to_align, intermediates, suppl)
        ref.SetProp("_Name", f'{i}PA')
        to_align.SetProp("_Name", f'{i}PB')

        for j in range(len(intermediates)):
            intermediates[j].SetProp("_Name", f'{i}I{j}')

        # write to file
        if sdf:
            with Chem.SDWriter(f'rbfe_gromacs/input/ligands/aligned_{i}.sdf') as w:
                w.write(to_align)
                for j in range(len(intermediates)):
                    w.write(intermediates[j])
                w.write(ref)
