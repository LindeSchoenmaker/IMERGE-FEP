import operator
from abc import ABC, abstractmethod

import pandas as pd
from rdkit import Chem

from rgroupinterm.utils.compute_score import computeLOMAPScore


class Pruner(ABC):
    """Abstract class for pruners."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out some rows from a dataframe.

        Args:
            df (pd.DataFrame): dataframe to be filtered

        Returns:
            The filtered pd.DataFrame
        """


class Scorer(ABC):
    """
    Abstract class for intermediate scorers.

    Attribute:
        score_type (type): is score is a bool, integer or float
    """

    @property
    @abstractmethod
    def score_type(self):
        pass

    @abstractmethod
    def __call__(self, intermediate, pair) -> pd.DataFrame:
        """Score intermediate by comparing to it's two parent molecules.

        Args:
           intermediate: molecule
           pair: list of 2 molecules

        Returns:
            Score
        """


class BasePruner(ABC):
    """
    Base class for pruners.

    Pruner will either return scored intermediates, the top n scoring intermediates, intermediates for which scoring returns true.
    
    Args:
        scorer (callable): class that has call that calculates score.
        transformer (callable, optional): class for transforming the score.
        topn (int, optional): how many of the best scoring intermediates to keep (pairwise if column pair is present, otherwise for whole dataframe)
        compare (callable, optional): operator for comparing score to threshold, by default greater equal, can be lt, le, eq, ne, ge, gt
        threshold (float, optional): keep intermediates that return for comparison
    """

    def __init__(self,
                 scorer: callable,
                 transformer: callable = None,
                 topn: int | None = None,
                 compare: callable = operator.ge,
                 threshold: float | None = None):
        self.scorer = scorer
        self.transformer = transformer
        # self.func_kwargs = func_kwargs if func_kwargs else {}
        self.topn = topn
        self.compare = compare
        self.threshold = threshold
        # add some checks, self.scorer.score_type cannot be bool with topn & threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get top N intermediated based on pruning criteria.

        Args:
            df (pd.DataFrame): dataframe to be filtered

        Returns:
            The filtered pd.DataFrame
        """
        # apply scoring function
        self.score_type = self.scorer.score_type
        df['score'] = df.apply(lambda row: self.scorer(row[
            'Intermediate'], [row['Parent_1'], row['Parent_2']]),
                               axis=1)
        # if transformer, transform score
        if self.transformer:
            df['score'] = df['score'].apply(self.transformer)
            self.score_type = self.transformer.score_type

        # do pruning
        if self.topn:
            if 'Pair' in df.columns:
                return df.groupby('Pair').apply(
                    lambda x: get_n_largest(x, self.topn))
            else:
                return df.apply(lambda x: get_n_largest(x, self.topn))
        if self.threshold:
            if self.score_type == bool:
                return df.loc[df['score']]
            else:
                return df.loc[df['score'].apply(
                    lambda x: self.compare(x, self.threshold))]
        else:
            return df


def get_n_largest(group, n):
    if len(group) < n:
        return group
    return group.loc[group['score'].nlargest(n).index]


class HeavyAtomScorer(Scorer):

    def __init__(self):
        self._score_type = float
        self.describe = 'smallest'

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, intermediate, pair):
        inputs = [intermediate]
        inputs.extend(pair)
        res = Chem.rdFMCS.FindMCS(inputs,
                                  matchValences=True,
                                  ringMatchesRingOnly=True,
                                  completeRingsOnly=True,
                                  timeout=2)
        core = Chem.MolFromSmarts(res.smartsString)
        res, _ = Chem.rdRGroupDecomposition.RGroupDecompose([core],
                                                            inputs,
                                                            asSmiles=False,
                                                            asRows=False)
        df = pd.DataFrame(res)
        interm_rs = df.iloc[0][[
            column for column in df.columns if column.startswith('R')
        ]].to_list()
        scores = [r.GetNumHeavyAtoms() for r in interm_rs]
        if self.describe == 'smallest':
            return min(scores)

        return scores

def GetNumHeavyAtoms(mol):
    return mol.GetNumHeavyAtoms()


class LomapScorer(Scorer):

    def __init__(self):
        self._score_type = float

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, intermediate, pair):
        liga = pair[0]
        ligb = pair[1]

        lomap_score_am = computeLOMAPScore(liga, intermediate)
        lomap_score_mb = computeLOMAPScore(intermediate, ligb)

        return lomap_score_am + lomap_score_mb
