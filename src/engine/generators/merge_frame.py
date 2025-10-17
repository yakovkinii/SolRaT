import inspect
import logging
from typing import List, Callable, Dict, Union

import pandas as pd

from src.engine.generators.merge_loopers import Looper, Dummy


def merge(df1, df2, on=None):
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    if len(on) == 0:
        return df1.merge(df2, how='cross')
    else:
        return df1.merge(df2, on=on, how='inner')


class Frame:
    class FrameFactor:
        def __init__(self, name:str, factor:Callable = None, dependencies:List[str]=None, merged:bool=False):
            self.name:str = name
            self.call:Callable = factor
            if dependencies is not None:
                self.dependencies: List[str] = dependencies
            else:
                assert factor is not None
                self.dependencies: List[str] = [p.name for p in inspect.signature(factor).parameters.values()]
            self.merged: bool = merged

    def __init__(self, base_frame:pd.DataFrame=None, **kwargs: Looper):
        if base_frame is not None:
            self.frame: pd.DataFrame = base_frame.copy()
        else:
            self.frame: pd.DataFrame = pd.DataFrame(index=[0], columns=[])

        for looper_name, looper in kwargs.items():
            looper.set_name(looper_name)
            if isinstance(looper, Dummy):
                continue
            dependent_cols = list(looper.get_directly_dependent_columns())
            sub_frame = self.construct_sub_frame(dependent_cols)
            sub_frame_filled = looper.fill_frame(sub_frame)
            assert not sub_frame_filled[looper_name].isna().any()
            self.frame = merge(self.frame, sub_frame_filled)

        self.factors: Dict[str, Frame.FrameFactor] = {}
        logging.info(f"Frame shape after initialization: {self.frame.shape}")

    def copy(self):
        new_frame = Frame()
        new_frame.frame = self.frame.copy()
        new_frame.factors = {name: Frame.FrameFactor(factor.name, factor.call, factor.dependencies.copy(), factor.merged) for name, factor in self.factors.items()}
        return new_frame

    def construct_sub_frame(self, columns:List[str])->pd.DataFrame:
        if len(columns) == 0:
            return pd.DataFrame(index=[0], columns=[])
        return self.frame[columns].drop_duplicates().reset_index(drop=True)

    def set_factors(self, **kwargs:Callable):
        for name, factor_callable in kwargs.items():
            self.factors[name] = self.FrameFactor(name, factor_callable)

    def get_dependent_factors(self, column:str)->List[str]:
        return [name for name, factor in self.factors.items() if column in factor.dependencies]

    def get_merged_independent_factors(self, column:str)->List[str]:
        return [name for name, factor in self.factors.items() if factor.merged and column not in factor.dependencies]

    def merge_factor(self, factor_name:str):
        logging.info(f"Merging factor {factor_name}")
        factor = self.factors[factor_name]
        factor_frame = self.construct_sub_frame(factor.dependencies)
        arguments = {name: factor_frame[name].values.reshape(-1, 1) for name in factor.dependencies}
        factor_frame[factor_name] = factor.call(**arguments)
        self.frame = merge(self.frame, factor_frame)
        factor.merged = True

    def combine_merged_factors(self, factor_names:List[str]):
        if len(factor_names) == 0:
            raise ValueError("Trying to combine zero factors. No factors depend on a summation coefficient?")

        if len(factor_names) == 1:
            return factor_names[0]

        new_factor_name = "*".join(factor_names)
        self.frame[new_factor_name] = self.frame[factor_names].prod(axis=1)
        dependencies = list(set().union(*[self.factors[name].dependencies for name in factor_names]))
        self.factors[new_factor_name] = self.FrameFactor(new_factor_name, dependencies=dependencies, merged=True)

        for factor_name in factor_names:
            del self.frame[factor_name]
            del self.factors[factor_name]

        return new_factor_name

    def remove_dependency(self, column:str):
        for factor in self.factors.values():
            if column in factor.dependencies:
                factor.dependencies.remove(column)

    def get_other_frame_columns(self, exclude:str)->List[str]:
        return [col for col in self.frame.columns if col != exclude and col not in self.factors]

    def reduce_single_index(self, column: Union[str, Looper]):
        if isinstance(column, Looper):
            column = column.get_name()
        dependent_factors = self.get_dependent_factors(column)
        for factor_name in dependent_factors:
            if not self.factors[factor_name].merged:
                self.merge_factor(factor_name)

        factor_name = self.combine_merged_factors(dependent_factors)
        merged_independent_factors = self.get_merged_independent_factors(column)
        assert self.get_dependent_factors(column) == [factor_name]

        self.remove_dependency(column)

        group_columns = self.get_other_frame_columns(column)
        if len(group_columns) == 0:
            assert len(merged_independent_factors) == 0, "There are independent merged factors: the frame can be factorized."
            self.frame = self.frame.drop(columns=column)
            return self.frame[factor_name].sum()

        self.frame = self.frame.groupby(group_columns).agg(
            {factor_name: 'sum',
             **{f: 'first' for f in merged_independent_factors},
             }
        )
        self.frame = self.frame.reset_index()
        return None

    def _reduce(self, columns):
        result = None
        for col in columns:
            assert col not in self.factors
            assert col in self.frame.columns
            result = self.reduce_single_index(col)
        return result

    def reduce(self, *args: Union[Looper, str]):
        """ usage:
        frame.reduce() to reduce all,
        frame.reduce(col1, col2, ..., col5, col6) to specify first and last columns to reduce
        """
        factor_columns = list(self.factors.keys())

        if len(args) == 0 or (len(args) == 1 and args[0] is Ellipsis):
            return self._reduce(
                [col for col in self.frame.columns[::-1] if col not in factor_columns]
            )
        if Ellipsis not in args:
            result = self._reduce([col.get_name() if isinstance(col, Looper) else col for col in args])
            if result is None:
                logging.warning("The frame is not fully reduced. Consider using Ellipsis (...) "
                                "to reduce all remaining columns: frame.reduce(col1, col2, ...).")
            return result

        if args.count(Ellipsis) > 1:
            raise ValueError("Only one Ellipsis (...) is allowed in reduce() arguments.")

        ellipsis_index = args.index(Ellipsis)
        columns_before = [col.get_name() if isinstance(col, Looper) else col for col in args[:ellipsis_index]]
        columns_after = [col.get_name() if isinstance(col, Looper) else col for col in args[ellipsis_index + 1:]]

        frame_columns = [col for col in self.frame.columns if col not in factor_columns]
        frame_columns = [col for col in frame_columns if col not in columns_before + columns_after]
        frame_columns = columns_before + frame_columns + columns_after
        return self._reduce(frame_columns)

    def debug_print_structure(self):
        print("----- Frame -----")
        print("Current Frame Structure:")
        print(self.frame)
        print("Unmerged factors:")
        for name, factor in self.factors.items():
            if not factor.merged:
                print(f" - {name}: depends on {factor.dependencies}")
        print("-----------------")

    def debug_evaluate_legacy(self):
        for factor_name in list(self.factors.keys()):
            self.merge_factor(factor_name)
        factor_names = list(self.factors.keys())
        return self.frame[factor_names].prod(axis=1).sum()