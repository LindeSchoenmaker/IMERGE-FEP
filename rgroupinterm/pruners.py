import operator
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem

from rgroupinterm.utils.compute_score import computeLOMAPScore, computeROCSScore


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

    def __init__(self, transformer: Callable = None):
        self.transformer = transformer

    @property
    @abstractmethod
    def score_type(self):
        pass

    def __call__(self, intermediate, pair) -> pd.DataFrame:
        """Score intermediate and apply transform.

        Args:
           intermediate: molecule
           pair: list of 2 molecules

        Returns:
            Score: (transformed) score
        """
        scores = self.calc_score(intermediate, pair)
        if self.transformer:
            return self.transformer(scores)
        return scores

    @abstractmethod
    def calc_score(self, intermediate, pair) -> pd.DataFrame:
        """Score intermediate by comparing to it's two parent molecules.

        Args:
           intermediate: molecule
           pair: list of 2 molecules

        Returns:
            Score: raw score(s)
        """


class Transformer(ABC):
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
        """Score intermediate and apply transform.

        Args:
           intermediate: molecule
           pair: list of 2 molecules

        Returns:
            Score: (transformed) score
        """


class BasePruner(ABC):
    """
    Base class for pruners.

    Pruner will either return scored intermediates, the top n scoring intermediates, intermediates for which scoring returns true.
    
    Args:
        scorer (Callable): class that has call that calculates score.
        transformer (Callable, optional): class for transforming the score.
        topn (int, optional): how many of the best scoring intermediates to keep (pairwise if column pair is present, otherwise for whole dataframe)
        compare (Callable, optional): operator for comparing score to threshold, by default greater equal, can be lt, le, eq, ne, ge, gt
        threshold (float, optional): keep intermediates that return for comparison
    """

    def __init__(self,
                 scorers: list[Callable],
                 transformers: list[Callable] = None,
                 topn: int | None = None,
                 compare: Callable = operator.ge,
                 threshold: float | None = None):
        self.scorers = scorers
        self.transformers = transformers
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
        i = 0
        for scorer in self.scorers:
            #TODO save score_type as dict
            self.score_type = scorer.score_type
            df_scores = df.apply(lambda row: scorer(row[
                'Intermediate'], [row['Parent_1'], row['Parent_2']]),
                                 axis=1,
                                 result_type='expand')
            df_scores, i = rename_columns(df_scores, i)
            df = pd.concat([df, df_scores], axis=1)

        # if transformer, transform score
        raw_columns = [
            column for column in df.columns if column.startswith('raw_score')
        ]
        if self.transformers:
            i = 0
            for transformer in self.transformers:
                if transformer.__class__ == 'NormalizeTransformer':
                    return df.groupby('Pair').apply(transformer)
                else:
                    df_trans = df[raw_columns].apply(transformer, axis=1)
                df_trans, i = rename_columns(df_trans,
                                                i,
                                                name='trans_score')
                self.score_type = transformer.score_type
                df = pd.concat([df, df_trans], axis=1)
            trans_columns = [
                column for column in df.columns
                if column.startswith('trans_score')
            ]
            df['score'] = df[trans_columns[-1]]
        elif len(raw_columns) == 1:
            df.rename(columns={'raw_score_0': 'score'}, inplace=True)

        # do pruning
        if self.topn or self.threshold:
            #TODO update so that works if multiple transforms or transforming doesn't yield one column
            if len(raw_columns) > 1 and self.transformers is None:
                raise ValueError(
                    'multiple scores are returned, requires transformation before molecules can be pruned'
                )
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


def rename_columns(df, i, name = 'raw_score'):
    if isinstance(df, pd.Series):
        df.name = f'{name}_{i}'
        i += 1
    elif isinstance(df, pd.DataFrame):
        df.columns = [
            f'{name}_{i+x}' for x in range(len(df.columns))
        ]
        i += len(df.columns)
    return df, i

class HeavyAtomScorer(Scorer):

    def __init__(self, transformer: Callable = None):
        super(HeavyAtomScorer, self).__init__(transformer)
        self._score_type = float
        self.describe = 'smallest'

    @property
    def score_type(self):
        return self._score_type

    def calc_score(self, intermediate, pair):
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

        return scores

def GetNumHeavyAtoms(mol):
    return mol.GetNumHeavyAtoms()


class LomapScorer(Scorer):

    def __init__(self, transformer: Callable = None):
        super(LomapScorer, self).__init__(transformer)
        self._score_type = float

    @property
    def score_type(self):
        return self._score_type

    def calc_score(self, intermediate, pair):
        liga = pair[0]
        ligb = pair[1]

        lomap_score_am = computeLOMAPScore(liga, intermediate)
        lomap_score_mb = computeLOMAPScore(intermediate, ligb)

        return lomap_score_am, lomap_score_mb


class ROCSScorer(Scorer):

    def __init__(self, transformer: Callable = None):
        super(ROCSScorer, self).__init__(transformer)
        self._score_type = float

    @property
    def score_type(self):
        return self._score_type

    def calc_score(self, intermediate, pair):
        liga = pair[0]
        ligb = pair[1]

        rocs_score_am = computeROCSScore(liga, intermediate)
        rocs_score_mb = computeROCSScore(intermediate, ligb)

        return rocs_score_am, rocs_score_mb


class SumTransformer(Transformer):

    def __init__(self):
        self._score_type = float

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, scores):
        return sum(scores)


class MinTransformer(Transformer):

    def __init__(self):
        self._score_type = float

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, scores):
        return min(scores)


class HarmonicMeanTransformer(Transformer):

    def __init__(self, exponent: int = 1):
        self._score_type = float
        self.exponent = exponent

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, scores):
        assert len(scores) == 2
        score = 2 * (scores[0]**self.exponent * scores[1]**self.exponent) / (
            scores[0]**self.exponent + scores[1]**self.exponent)
        return score


class WeightedSumTransformer(Transformer):

    def __init__(self, weights: list):
        self._score_type = float
        self.weights = weights

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, scores):
        assert len(scores) == len(self.weights)
        return np.sum(np.multiply(scores, self.weights))


class NormalizeTransformer(Transformer):

    def __init__(self):
        self._score_type = float

    @property
    def score_type(self):
        return self._score_type

    def __call__(self, group: pd.DataFrame): #check if series could be given
        return group
