import operator
from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition

from rgroupinterm.utils.compute_score import (
    computeLOMAPScore,
    computeROCSScore,
    computeTanimotoScore,
)


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
        score_prefix (str): name of column suffix specific to the scorer class
    """

    def __init__(self, transformer: Callable = None):
        self.transformer = transformer

    @property
    @abstractmethod
    def score_type(self):
        pass

    @property
    @abstractmethod
    def score_suffix(self):
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
        score_prefix (str): name of column suffix specific to the transformer class
    """
    @property
    @abstractmethod
    def score_type(self):
        pass

    @property
    @abstractmethod
    def score_suffix(self):
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
        self.score_type = None
        # add some checks, self.scorer.score_type cannot be bool with topn & threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get top N intermediated based on pruning criteria.

        Args:
            df (pd.DataFrame): dataframe to be filtered

        Returns:
            The filtered pd.DataFrame
        """
        # apply scoring function

        output_columns = []
        for scorer in self.scorers:
            #TODO save score_type as dict
            self.score_type = scorer.score_type
            df_scores = df.apply(lambda row: scorer(row[
                'Intermediate'], [row['Parent_1'], row['Parent_2']]),
                                 axis=1,
                                 result_type='expand')
            df_scores, names = rename_columns(df_scores,
                                              scorer.score_suffix,
                                              prefix='raw')
            output_columns.extend(names)
            df = pd.concat([df, df_scores], axis=1)
        if len(output_columns) == 0:
            output_columns = [x for x in df.columns if 'raw' in x]
        # if transformer, transform score
        if self.transformers:
            for transformer in self.transformers:
                # fix column names & which ones areused as input for the next
                if transformer.__class__ == NormalizeTransformer:
                    output_columns.append('Pair')
                    df_trans = df[output_columns].groupby('Pair').apply(
                        transformer)[output_columns[:-1]]
                    df_trans.index = df_trans.index.droplevel(0)
                else:
                    df_trans = df[output_columns].apply(transformer, axis=1)
                df_trans, names = rename_columns(df_trans,
                                                 transformer.score_suffix,
                                                 prefix='trans')
                self.score_type = transformer.score_type
                df = pd.concat([df, df_trans], axis=1)
                output_columns = names

        if len(output_columns) == 1:
            df['score'] = df[output_columns]
        # do pruning
        if self.topn or self.threshold:
            #TODO update so that works if multiple transforms or transforming doesn't yield one column
            if len(output_columns) > 1 and self.transformers is None:
                raise ValueError(
                    'multiple scores are returned, requires transformation before molecules can be pruned'
                )
            if self.topn:
                if 'Pair' in df.columns:
                    return df.groupby('Pair').apply(
                        lambda x: get_n_largest(x, self.topn))
                else:
                    #TODO look into why turns into series
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


def rename_columns(df, suffix, prefix = 'raw'):
    if isinstance(df, pd.Series):
        names = [f'{prefix}_score_{suffix}']
        df.name = names[0]
    elif isinstance(df, pd.DataFrame):
        names = [
            f'{prefix}_score_{suffix}_{x}' for x in range(len(df.columns))
        ]
        df.columns = names
    return df, names

class HeavyAtomScorer(Scorer):
    """
    Heavy atom scorer

    Attribute:
        score_type (type): is score is a bool, integer or float
        score_prefix (str): name of column suffix specific to the scorer class
        level (str): whether to return the number of heavy atoms in intermediate & parents (returns 3 scores)
                     or to return the number of heavy atoms of the r-groups of the intermediate (num scores equal to number of 3-groups)
    """

    def __init__(self, transformer: Callable = None, level: Literal['mol', 'rgroup'] = 'mol'):
        super(HeavyAtomScorer, self).__init__(transformer)
        self._score_type = float
        self._score_suffix = f'HeavyAtom_{level}'
        self.level = level

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def calc_score(self, intermediate, pair):
        if self.level == 'rgroup':
            inputs = [intermediate]
            inputs.extend(pair)
            res = Chem.rdFMCS.FindMCS(inputs,
                                    matchValences=True,
                                    ringMatchesRingOnly=True,
                                    completeRingsOnly=True,
                                    timeout=2)
            core = Chem.MolFromSmarts(res.smartsString)
            res, _ = rdRGroupDecomposition.RGroupDecompose([core],
                                                                inputs,
                                                                asSmiles=False,
                                                                asRows=False)
            df = pd.DataFrame(res)
            mol_obs = df.iloc[0][[
                column for column in df.columns if column.startswith('R')
            ]].to_list()
        elif self.level == 'mol':
            mol_obs = [intermediate]
            mol_obs.extend(pair)

        scores = [r.GetNumHeavyAtoms() for r in mol_obs]

        return scores

def GetNumHeavyAtoms(mol):
    return mol.GetNumHeavyAtoms()


class TanimotoScorer(Scorer):

    def __init__(self, transformer: Callable = None):
        super(TanimotoScorer, self).__init__(transformer)
        self._score_type = float
        self._score_suffix = 'Tanimoto'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def calc_score(self, intermediate, pair):
        liga = pair[0]
        ligb = pair[1]

        tanimoto_score_am = computeTanimotoScore(liga, intermediate)
        tanimoto_score_mb = computeTanimotoScore(intermediate, ligb)

        return tanimoto_score_am, tanimoto_score_mb


class LomapScorer(Scorer):

    def __init__(self, transformer: Callable = None):
        super(LomapScorer, self).__init__(transformer)
        self._score_type = float
        self._score_suffix = 'Lomap'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

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
        self._score_suffix = 'ROCS'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def calc_score(self, intermediate, pair):
        liga = pair[0]
        ligb = pair[1]

        rocs_score_am = computeROCSScore(liga, intermediate)
        rocs_score_mb = computeROCSScore(intermediate, ligb)

        return rocs_score_am, rocs_score_mb


class SumTransformer(Transformer):

    def __init__(self):
        self._score_type = float
        self._score_suffix = 'Sum'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, scores):
        return sum(scores)


class MinTransformer(Transformer):

    def __init__(self):
        self._score_type = float
        self._score_suffix = 'Min'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, scores):
        return min(scores)


class HarmonicMeanTransformer(Transformer):

    def __init__(self, exponent: int = 1):
        self._score_type = float
        self._score_suffix = 'HarmonicMean'
        self.exponent = exponent

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, scores):
        assert len(scores) == 2
        score = 2 * (scores[0]**self.exponent * scores[1]**self.exponent) / (
            scores[0]**self.exponent + scores[1]**self.exponent)
        return score


class WeightedSumTransformer(Transformer):

    def __init__(self, weights: list):
        self._score_type = float
        self._score_suffix = 'WeightedSum'
        self.weights = weights

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, scores):
        assert len(scores) == len(self.weights)
        return np.sum(np.multiply(scores, self.weights))


class NormalizeTransformer(Transformer):

    def __init__(self):
        self._score_type = float
        self._score_suffix = 'Normalized'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, df: pd.DataFrame):  #check if series could be given
        return (df - df.min()) / (df.max() - df.min())


class OperatorTransformer(Transformer):

    def __init__(self, a: int, b: int, compare: Callable = operator.ge):
        self._score_type = bool
        self._score_suffix = 'Operator'
        self.a = a
        self.b = b
        self.compare = compare

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, scores):
        return self.compare(scores[self.a], scores[self.b])


class SmallerThanTransformer(Transformer):
    """returns true if the first score is lower than at least one of other two scores"""

    def __init__(self):
        self._score_type = bool
        self._score_suffix = 'SmallerThan'

    @property
    def score_type(self):
        return self._score_type

    @property
    def score_suffix(self):
        return self._score_suffix

    def __call__(self, scores):
        if scores[0] <= scores[1] or scores[0] <= scores[2]:
            return True
        return False