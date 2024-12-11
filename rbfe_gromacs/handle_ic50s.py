print('start loading')

import json

import yaml
from rdkit import Chem

print('loaded')
if __name__ == "__main__":
    measured_mols = []
    print('starting')
    with open('rbfe_gromacs/ligands.yml', 'r') as file:
        reference_dict = yaml.safe_load(file)
    for key, value in reference_dict.items():
        mol = Chem.MolFromSmiles(value['smiles'])
        measured_mols.append(mol)
    # iterate over SMILES in rbfe calculations
    with open('rbfe_gromacs/input/rhfe_pairs.json', 'r') as file:
        pairs = json.load(file)
    for key, value in pairs.items():
        print(key)
        mol = Chem.MolFromSmiles(value)
        subms = [x for x in measured_mols if x.HasSubstructMatch(mol)]
        if len(subms) > 0:
            print([i for i, x in enumerate(measured_mols) if x.HasSubstructMatch(mol)])
        identical = [x for x in subms if mol.HasSubstructMatch(x)]
        print(len(identical))
        if len(identical) > 0:
            print(Chem.MolToSmiles(identical[0]))
