import glob
import operator
import os
import unittest  # The test framework
from itertools import combinations

import pandas as pd
from rdkit import Chem, rdBase

from rgroupinterm.pruners import (
    BasePruner,
    HarmonicMeanTransformer,
    HeavyAtomScorer,
    LomapScorer,
    MinTransformer,
    NormalizeTransformer,
    ROCSScorer,
    SmallerThanTransformer,
    SumTransformer,
    TanimotoScorer,
    Transformer,
    WeightedSumTransformer,
)
from rgroupinterm.rgroupenumeration import EnumRGroups

rdBase.DisableLog('rdApp.*')
os.environ["OE_LICENSE"] = "/home/linde/.OpenEye/oe_license.txt"


class Test_EnumerateRGroups(unittest.TestCase):

    def test_enumerate_chirality_1R(self):
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
            'FC(F)(c1cc([C@H]2OCCC[C@H]2[C@@H](N3)c4ccc(O)cc4)c3cc1)F',
            '[NH3+]C[C@H]1CC[C@@H]2[C@@H](c3c(N[C@H]2c4ccc(O)cc4)ccc(C(C)(C)C)c3)O1',
            '[NH3+]C[C@H]1CC[C@@H]2[C@@H](c3c(N[C@H]2c4ccc(O)cc4)ccc(C(F)(F)F)c3)O1',
            'CC(C)(C)c1ccc2c(c1)[C@H]1OCCC[C@H]1[C@H](c1ccccc1)N2',
            'FC(F)(c1cc([C@H]2OCCC[C@H]2[C@@H](N3)c4ccccc4)c3cc1)F',
            '[NH3+]C[C@H]1CC[C@@H]2[C@@H](c3c(N[C@H]2c4ccccc4)ccc(C(C)(C)C)c3)O1'
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

        nonchiral = Chem.MolFromSmiles(
            'CC(C)(c1cc(C2OCCCC2C(N3)c4ccccc4)c3cc1)C')
        subms_nonchiral = [
            self.same_mol(x, nonchiral) for x in generated_interm
        ]

        self.assertTrue(None not in generated_interm)
        self.assertEqual(len(generated_interm),
                         len(expected_interm))  # check length
        self.assertEqual(
            sum(subms_expected), 6)  # check if generated intermediates are same as expected
        self.assertEqual(subms_unexpected,
                        [False] * 6)  # check if previous test is valid
        self.assertEqual(subms_nonchiral,
                        [False] * 6)  # check if chiral information captured by test

    def same_mol(self, mol1, mol2):
        return mol1.HasSubstructMatch(mol2, useChirality=True) and mol2.HasSubstructMatch(mol1, useChirality=True)

    def test_enumerate_eg5(self):
        # get molecule pairs
        dir = 'eg5'
        path = "/zfsdata/data/linde/fep_intermediate_generation/ligands/"
        addendum = '*.sdf'
        mols = []
        for file in glob.glob(path + dir + '/' + addendum):
            mol = Chem.rdmolfiles.SDMolSupplier(file)[0]
            mols.append(mol)

        for i, (liga, ligb) in enumerate(combinations(mols, 2)):
            generator = EnumRGroups()
            liga = Chem.Mol(liga)
            ligb = Chem.Mol(ligb)
            df_interm, core = generator.generate_intermediates([liga, ligb])
            generated_interm = df_interm['Intermediate'].tolist()
            self.assertTrue(None not in generated_interm)
            self.assertEqual(len(generated_interm),
                             2**(len(generator.columns)) - 2)

    def test_multiple_core_matches(self):
        generator = EnumRGroups()
        smiles = [
            'C[NH2+]C[C@H]1CC[C@@H]2[C@H](O1)c1cc(C(F)(F)F)ccc1N[C@H]2c1ccc(F)cc1',
            'CC[N@H+](CCO)C[C@H]1CC[C@@H]2[C@H](O1)c1cc(C(C)(C)C)ccc1N[C@H]2c1ccccc1'
        ]
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        df_interm, core = generator.generate_intermediates(mols)
        generated_interm = df_interm['Intermediate'].tolist()
        self.assertTrue(None not in generated_interm)
        self.assertEqual(len(generated_interm),
                         2**(len(generator.columns)) - 2)

        # test chirality applied correctly for one of the intermediates
        expected_intermediate = Chem.MolFromSmiles('CC[N@@H+](C[C@H]1CC[C@@H]2[C@@H](c3c(N[C@H]2c4ccccc4)ccc(C(C)(C)C)c3)O1)C')
        subms = [
            self.same_mol(x, expected_intermediate) for x in generated_interm
        ]
        self.assertEqual(sum(subms), 1)


class dummy_scorer():
    def __init__(self):
        self._score_type = int
        self._score_suffix = 'test'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, intermediate, pair: list, add1=False):
        a = 1
        if add1:
            a += 1
        return a

class transformer(Transformer):
    def __init__(self):
        self._score_type = bool
        self._score_suffix = 'test'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, score):
        return score > 1.2


class Test_Pruners(unittest.TestCase):

    def test_base(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']

        pruner = BasePruner([dummy_scorer()], None)
        pruned_df = pruner(df_mols)
        self.assertTrue('score' in pruned_df.columns)

        pruner = BasePruner([dummy_scorer()], None, threshold=1.2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 0)

        pruner = BasePruner([dummy_scorer()],
                            None,
                            threshold=1.2,
                            compare=operator.le)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 10)

        pruner = BasePruner([dummy_scorer()], None, threshold=0.8)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 10)

        pruner = BasePruner([dummy_scorer()], [transformer()])
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 10)

        pruner = BasePruner([dummy_scorer()], [transformer()], threshold=True)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 0)

    def test_tanimoto(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']

        pruner = BasePruner(
            [TanimotoScorer(transformer=HarmonicMeanTransformer(exponent=4))],
            topn=2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 6)

    def test_lomap(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']

        pruner = BasePruner([LomapScorer(transformer=SumTransformer())],
                            topn=1)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 3)

        pruner = BasePruner([LomapScorer(transformer=SumTransformer())],
                            topn=2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 6)

        pruner = BasePruner(
            [LomapScorer(transformer=HarmonicMeanTransformer(exponent=4))],
            topn=2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 6)

    def test_rocs(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']

        pruner = BasePruner(
            [ROCSScorer(transformer=HarmonicMeanTransformer(exponent=2))],
            topn=2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 6)

    def test_metric_Daan_2D(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']
        pruner = BasePruner(
            [
                LomapScorer(transformer=HarmonicMeanTransformer()),
                TanimotoScorer(transformer=HarmonicMeanTransformer(exponent=4))
            ],
            transformers=[
                NormalizeTransformer(),
                WeightedSumTransformer(weights=[0.2, 0.8])
            ],  #TODO would be nicer if weights more clearly coupled to score(r)
            topn=2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 6)

    def test_metric_Daan_3D(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']
        pruner = BasePruner(
            [
                LomapScorer(transformer=HarmonicMeanTransformer()),
                ROCSScorer(transformer=HarmonicMeanTransformer(exponent=2))
            ],
            transformers=[
                NormalizeTransformer(),
                WeightedSumTransformer(weights=[0.2, 0.8])
            ],  #TODO would be nicer if weights more clearly coupled to score(r)
            topn=2)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 6)

    def test_heavyatom(self):
        df = pd.read_csv('data/eg_5_intermediates.csv')[:10]
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']

        pruner = BasePruner([HeavyAtomScorer(MinTransformer(), level='rgroup')], threshold=1)
        pruned_df = pruner(df_mols)
        self.assertEqual(len(pruned_df), 3)

        pruner = BasePruner([HeavyAtomScorer(SmallerThanTransformer(), level='mol')])
        pruned_df = pruner(df_mols)

if __name__ == '__main__':
    unittest.main()
