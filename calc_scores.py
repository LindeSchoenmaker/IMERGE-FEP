import os

import pandas as pd
from rdkit import Chem

from rgroupinterm.pruners import (
    BasePruner,
    HarmonicMeanTransformer,
    LomapScorer,
    ROCSScorer,
    SumTransformer,
    TanimotoScorer,
)
from rgroupinterm.utils.compute_score import (
    computeLOMAPScore,
    computeROCSScore,
    computeTanimotoScore,
)


def write_list(path, idxs):
    with open(path, 'w') as fp:
        for item in idxs:
            fp.write("%s\n" % item)


def get_harmonicmean(df):
    for score in ['Tanimoto', 'Lomap', 'ROCS']:
        df_score = df[[x for x in df.columns if score in x]]
        for transformer in [HarmonicMeanTransformer()]:
            print(transformer)
            pruner = BasePruner([], [transformer])
            scored_df = pruner(df_score)
            scored_df['Pair'] = df['Pair']
            scored_df['Set'] = df['Set']
            print(scored_df['score'].mean())
            print(scored_df.groupby(['Set','Pair']).mean()['score'].mean())
            print(scored_df.groupby(['Set','Pair']).min()['score'].mean())
            print(scored_df.groupby(['Set','Pair']).max()['score'].mean())


def get_average(df):
    # get average
    for score in ['Tanimoto', 'Lomap', 'ROCS']:
        df_score = df[[x for x in df.columns if score in x]]
        print(score)
        for transformer in [SumTransformer(), HarmonicMeanTransformer()]:
            print(transformer)
            pruner = BasePruner([], [transformer])
            scored_df = pruner(df_score)
            scored_df['Pair'] = df['Pair']
            print(scored_df[f'trans_score_{transformer.score_suffix}'].mean())
            print(scored_df.groupby('Pair').mean()[f'trans_score_{transformer.score_suffix}'].mean())


def keep_unique(name = 'jacs', targetset=None):
    df = pd.read_csv(f'data/{name}_intermediates.csv')
    df_unique = pd.DataFrame()
    if targetset:
        df = df.loc[df['Set'] == targetset]
    for subset in df.Set.unique():
        for pair in df.Pair.unique():
            df_temp = df.loc[(df['Set'] == subset)
                            & (df['Pair'] == pair)]
            intermediates = df_temp.Intermediate.to_list()
            canonical = [
                Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in intermediates
            ]
            if len(canonical) > len(set(canonical)):
                df_temp['Intermediate'] = df_temp['Intermediate'].apply(
                    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
                df_temp = df_temp.drop_duplicates(subset=['Intermediate'])
                df_unique = pd.concat([df_unique, df_temp])
            else:
                df_unique = pd.concat([df_unique, df_temp])
    df_unique.to_csv(f'data/{name}{targetset if targetset else ""}_intermediates_unique.csv', index=False)

def obtain_scored_intermediates(name = 'jacs', targetset=None):
    if not os.path.exists(f'data/{name}{targetset if targetset else ""}_intermediates_unique.csv'):
        keep_unique(name, targetset)
    df = pd.read_csv(f'data/{name}{targetset if targetset else ""}_intermediates_unique.csv')
    if not os.path.exists(f'data/{name}{targetset if targetset else ""}_scored_new.csv'):
        df_mols = pd.DataFrame()
        for column in ['Intermediate', 'Parent_1', 'Parent_2']:
            df_mols[column] = df[column].apply(lambda x: Chem.MolFromSmiles(x))
        df_mols['Pair'] = df['Pair']
        df_mols['Set'] = df['Set']

        pruner = BasePruner([LomapScorer(), ROCSScorer(),TanimotoScorer()]) 
        scored_df = pruner(df_mols)
        scored_df['Intermediate'] = df['Intermediate']
        scored_df['Parent_1'] = df['Parent_1']
        scored_df['Parent_2'] = df['Parent_2']
        scored_df.to_csv(f'data/{name}{targetset if targetset else ""}_scored_new.csv', index=False)

def obtain_scored_parents(name = 'jacs', targetset=None):
    df = pd.read_csv(f'data/{name}_intermediates.csv')
    if not os.path.exists(f'data/{name}{targetset if targetset else ""}_score_parents_new.csv'):
        if targetset:
            df = df.loc[df['Set'] == targetset]
        df_p = df.drop_duplicates(subset=['Set', 'Pair'])
        df_mols = pd.DataFrame()
        for column in ['Parent_1', 'Parent_2']:
            df_mols[column] = df_p[column].apply(
                lambda x: Chem.MolFromSmiles(x))
        df_mols['Set'] = df_p['Set']
        df_mols['Pair'] = df_p['Pair']
        df_mols['raw_score_Tanimoto_p'] = df_mols.apply(
            lambda row: computeTanimotoScore(row['Parent_1'], row['Parent_2']),
            axis=1)
        df_mols['raw_score_Lomap_p'] = df_mols.apply(
            lambda row: computeLOMAPScore(row['Parent_1'], row['Parent_2']),
            axis=1)
        df_mols['raw_score_ROCS_p'] = df_mols.apply(
            lambda row: computeROCSScore(row['Parent_1'], row['Parent_2']),
            axis=1)
        df_mols['Parent_1'] = df_p['Parent_1']
        df_mols['Parent_2'] = df_p['Parent_2']
        df_mols.to_csv(f'data/{name}{targetset if targetset else ""}_score_parents_new.csv', index=False)


if __name__ == "__main__":
    obtain_scored_intermediates(name = 'jacs')
    obtain_scored_parents(name = 'jacs')
