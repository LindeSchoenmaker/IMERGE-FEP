import glob
import time
from itertools import combinations

import pandas as pd
from rdkit import Chem

from rgroupinterm.rgroupenumeration import EnumRGroups

if __name__ == "__main__":
    df_comb = pd.DataFrame()
    folders = [folder.split('/')[-1] for folder in glob.glob("fep_intermediate_generation/ligands/*")]
    failed_combs = []
    runtimes = []
    for folder in folders:
        if folder in ['readme.txt']: 
            continue
        print(folder)
        path = "fep_intermediate_generation/ligands/"
        addendum = '*.sdf'
        mols = []
        for file in glob.glob(path + folder + '/' + addendum):
            mol = Chem.rdmolfiles.SDMolSupplier(file)[0]
            mols.append(mol)

        for i, (liga, ligb) in enumerate(combinations(mols, 2)):
            start = time.time()
            generator = EnumRGroups()
            liga = Chem.Mol(liga)
            ligb = Chem.Mol(ligb)
            try:
                df_interm, core = generator.generate_intermediates([liga, ligb])
                if 'Intermediate' in df_interm.columns:
                    generated_interm = df_interm['Intermediate'].tolist()
                    
                    df_interm['Set'] = folder
                    df_interm['Pair'] = i
                    df_interm['Parent_1'] = Chem.MolToSmiles(liga)
                    df_interm['Parent_2'] = Chem.MolToSmiles(ligb)
                    df_comb = pd.concat([df_comb, df_interm], ignore_index=True)
            except:
                print(Chem.MolToSmiles(liga), Chem.MolToSmiles(ligb))
                failed_combs.append([Chem.MolToSmiles(x) for x in [liga, ligb]])
                pass
            end = time.time()
            runtimes.append(end-start)
            
    df_comb['Intermediate'] = df_comb['Intermediate'].apply(lambda x: Chem.MolToSmiles(x))
    df_comb.to_csv('data/jacs_intermediates.csv', index=False)

    df = pd.DataFrame(failed_combs)
    df.to_csv('data/jacs_intermediates_failing.csv')
    with open('runtimes.txt', 'w') as fp:
        for item in runtimes:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')