import unittest  # The test framework

import pandas as pd
from rdkit import Chem, rdBase

from rgroupinterm.rgroupenumeration import EnumRGroups

rdBase.DisableLog('rdApp.*')


class Test_EnumerateRGroups(unittest.TestCase):

    def test_enumerate(self):
        generator = EnumRGroups()
        smiles = [
            'CC(C)(C)c1ccc2c(c1)[C@H]1OCCC[C@H]1[C@H](c1ccc(O)cc1)N2',
            '[NH3+]C[C@H]1CC[C@@H]2[C@H](O1)c1cc(C(F)(F)F)ccc1N[C@H]2c1ccccc1'
        ]
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        df_interm, core = generator.generate_intermediates(mols)

        self.assertTrue(isinstance(df_interm, pd.DataFrame))
        self.assertTrue(isinstance(core, Chem.rdchem.Mol))

        expected_interm = [
            'Oc1ccc(C2Nc3ccc(C(F)(F)F)cc3C3OCCCC23)cc1',
            'CC(C)(C)c1ccc2c(c1)C1OC(C[NH3+])CCC1C(c1ccc(O)cc1)N2',
            '[NH3+]CC1CCC2C(c3ccc(O)cc3)Nc3ccc(C(F)(F)F)cc3C2O1',
            'CC(C)(C)c1ccc2c(c1)C1OCCCC1C(c1ccccc1)N2',
            'FC(F)(F)c1ccc2c(c1)C1OCCCC1C(c1ccccc1)N2',
            'CC(C)(C)c1ccc2c(c1)C1OC(C[NH3+])CCC1C(c1ccccc1)N2'
        ]
        expected_interm = [Chem.MolFromSmiles(x) for x in expected_interm]

        generated_interm = df_interm['Intermediate'].tolist()

        subms_expected = [
            self.same_mol(generated_interm[x], expected_interm[x])
            for x in range(6)
        ]

        unexpected_interm = Chem.MolFromSmiles(
            'C(F)(F)c1ccc2c(c1)C1OCCCC1C(c1ccccc1)N2')
        subms_unexpected = [
            self.same_mol(x, unexpected_interm) for x in generated_interm
        ]

        self.assertTrue(None not in generated_interm)
        self.assertEqual(len(generated_interm), len(expected_interm)) # check length
        self.assertTrue(subms_expected, [True] * 6) # check if generated intermediates are same as expected
        self.assertTrue(subms_unexpected, [False] * 6)

    def same_mol(self, mol1, mol2):
        return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

if __name__ == '__main__':
    unittest.main()
