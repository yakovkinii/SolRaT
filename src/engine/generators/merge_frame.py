"""
TODO
TODO  This file needs improved documentation.
TODO
"""

import inspect
import logging
from typing import List, Callable, Dict, Union

import numpy as np
import pandas as pd

from src.engine.generators.merge_loopers import Looper, DummyOrAlreadyMerged


def merge(df1, df2, on=None):
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    if len(on) == 0:
        return df1.merge(df2, how="cross")
    else:
        return df1.merge(df2, on=on, how="inner")


class SumLimits:
    @classmethod
    def get_indexes(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__") and not callable(v)}

    def __init__(self):
        indexes = self.get_indexes()
        for k,v in indexes.items():
            v.name = None


class Frame:
    class FrameFactor:
        def __init__(
            self,
            name: str,
            factor: Callable = None,
            dependencies: List[str] = None,
            merged: bool = False,
            elementwise: bool = False,
        ):
            self.name: str = name
            self.call: Callable = factor
            if dependencies is not None:
                self.dependencies: List[str] = dependencies
            else:
                assert factor is not None
                self.dependencies: List[str] = [p.name for p in inspect.signature(factor).parameters.values()]
            self.merged: bool = merged
            self.elementwise: bool = elementwise
            logging.info(f"Created: {self}")

        def __repr__(self):
            return f"FrameFactor {self.name}. Dependencies: {self.dependencies}. Merged: {self.merged}. Elementwise: {self.elementwise}"

    @staticmethod
    def from_sum_limits(base_frame: pd.DataFrame, sum_linits: type(SumLimits)):
        looper_dict = sum_linits.get_indexes()
        return Frame(base_frame=base_frame, **looper_dict)

    def __init__(self, base_frame: pd.DataFrame = None, **kwargs: Looper):
        """
        1. loopers are merged IMMEDIATELY to the base frame.
        2. factors are stored and evaluated+merged when needed.
        """
        if base_frame is not None:
            self.frame: pd.DataFrame = base_frame.copy()
        else:
            self.frame: pd.DataFrame = pd.DataFrame(index=[0], columns=[])

        for looper_name, looper in kwargs.items():
            looper.set_name(looper_name)
            if isinstance(looper, DummyOrAlreadyMerged):
                continue
            dependent_cols = list(looper.get_directly_dependent_columns())
            sub_frame = self.construct_sub_frame(dependent_cols)
            sub_frame_filled = looper.fill_frame(sub_frame)
            assert not sub_frame_filled[looper_name].isna().any()
            self.frame = merge(self.frame, sub_frame_filled)
            logging.info(f'Merged {looper_name}, frame shape = {self.frame.shape}')

        self.factors: Dict[str, Frame.FrameFactor] = {}
        self._n_factors = 0  # for naming only
        logging.info(f"Frame shape after initialization: {self.frame.shape}")

    def __repr__(self):
        result = "=" * 10 + "\n"
        result += "FRAME:\n"
        result += "=" * 10 + "\n"
        result += "head:\n"
        result += str(self.frame.head()) + "\n"
        result += "-" * 10 + "\n"
        result += f"shape: {self.frame.shape}\n"
        result += "-" * 10 + "\n"
        result += "factors:\n"
        result += "-" * 10 + "\n"
        for factor in self.factors:
            result += str(factor) + "\n"

        return result

    def copy(self):
        new_frame = Frame()
        new_frame.frame = self.frame.copy()
        new_frame.factors = {
            name: Frame.FrameFactor(factor.name, factor.call, factor.dependencies.copy(), factor.merged)
            for name, factor in self.factors.items()
        }
        return new_frame

    def construct_sub_frame(self, columns: List[str]) -> pd.DataFrame:
        """
        This is used to reduce the evaluations of loopers/factors to minimum:
        we get all unique dependencies, evaluate on them, then merge back to the frame.
        """
        if len(columns) == 0:
            return pd.DataFrame(index=[0], columns=[])
        return self.frame[columns].drop_duplicates().reset_index(drop=True)

    def add_factors_to_multiply(self, *args: Callable, elementwise: bool = False, **kwargs):
        """
        This just registers the factors. They will be evaluated/merged later on demand.
        """
        for factor_callable in args:
            name = f"factor_{self._n_factors}"
            assert name not in self.frame.columns, f"Cannot add {name} as a factor: name already used."
            self.factors[name] = self.FrameFactor(name, factor_callable, elementwise=elementwise)
            self._n_factors += 1

        for name, factor_callable in kwargs.items():
            assert name not in self.frame.columns, f"Cannot add {name} as a factor: name already used."
            self.factors[name] = self.FrameFactor(name, factor_callable, elementwise=elementwise)
            self._n_factors += 1

    def get_dependent_factors(self, column: str) -> List[str]:
        return [name for name, factor in self.factors.items() if column in factor.dependencies]

    # def get_merged_independent_factors(self, column: str) -> List[str]:
    #     return [name for name, factor in self.factors.items() if factor.merged and column not in factor.dependencies]

    def merge_factor(self, factor_name: str):
        """
        Construct factor frame, evaluate, and merge it to the main frame
        """
        factor = self.factors[factor_name]
        logging.info(f"..Merging factor: {factor}")

        factor_frame = self.construct_sub_frame(factor.dependencies)
        # Reshape the dependencies so that they support vector evals.
        arguments = {name: factor_frame[name].values.reshape(-1, 1) for name in factor.dependencies}
        if factor.elementwise:
            # Do it row-wise, because the factor does not support array inputs.
            factor_frame[factor_name] = np.nan
            dfs = []
            for i in range(factor_frame.shape[0]):
                row_arguments = {name: arguments[name][i, 0] for name in factor.dependencies}
                dfs.append(pd.DataFrame({factor_name: [factor.call(**row_arguments)]}))
            factor_frame[factor_name] = pd.concat(dfs, ignore_index=True)
        else:
            # Regular logic: just create a column with the factor name and evaluate
            factor_frame[factor_name] = factor.call(**arguments)
        self.frame = merge(self.frame, factor_frame)
        factor.merged = True

    def combine_all_merged_factors(self):
        """
        Multiply all merged factors so that the frame has a single combined merged factor.
        """
        factor_names = [name for name, factor in self.factors.items() if factor.merged]

        assert len(factor_names) > 0, "There are zero merged factors"

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

    def remove_dependency(self, column: str):
        for factor in self.factors.values():
            if column in factor.dependencies:
                assert factor.merged, "Trying to remove a column dependency from unmerged factor"
                factor.dependencies.remove(column)

    def get_other_frame_columns(self, exclude: str) -> List[str]:
        """Get looper columns other than the specified one"""
        return [col for col in self.frame.columns if col != exclude and col not in self.factors]

    def reduce_single_index(self, column: Union[str, Looper]):
        """
        Reduction is Looper-wise (this way it clearly follows the logic of 'summation' operation)
        """
        if isinstance(column, Looper):
            column = column.get_name()

        logging.info(f"====")
        logging.info(f"Reducing column {column}:")
        dependent_factors = self.get_dependent_factors(column)
        logging.info(f"Dependent factors: {dependent_factors}")

        if len(dependent_factors) == 0:
            logging.error(f"No dependent factors for column {column}, dropping it directly.")
            self.remove_dependency(column)
            self.frame = self.frame.drop(columns=column)
            return self.frame

        for factor_name in dependent_factors:
            logging.info(f"  Ensuring factor {factor_name} is merged for reduction.")
            if not self.factors[factor_name].merged:
                logging.info(f"    Merging factor {factor_name} now.")
                self.merge_factor(factor_name)

        factor_name = self.combine_all_merged_factors()
        logging.info(f"  Combined dependent factors into {factor_name} for reduction.")
        self.remove_dependency(column)

        group_columns = self.get_other_frame_columns(column)
        logging.info(f"  Grouping by columns: {group_columns} to reduce {column}.")

        if len(group_columns) == 0:
            logging.info("  Reduced the last looper!")
            assert len(self.factors) == 1, f"Reduced all loopers, but some factors remain: {self.factors}"
            # self.frame = self.frame.drop(columns=column)
            # logging.info(f"  No grouping columns left, returning sum of {factor_name}.")
            logging.info("Calculating the sum over the last looper and returning the result")
            return self.frame[factor_name].sum()

        # Otherwise, group by the loopers that will be needed in future, and summate over current looper
        # self.frame = self.frame.groupby(group_columns).agg(
        #     {
        #         factor_name: "sum",
        #          **{f: "first" for f in merged_independent_factors},
        # }
        # )
        # self.frame = self.frame.reset_index()

        self.frame = self.frame.groupby(group_columns)[factor_name].sum().reset_index()
        logging.info(f"  Reduced frame shape: {self.frame.shape}")
        return None

    def _reduce(self, columns):
        result = None
        for col in columns:
            assert col not in self.factors, f"Reduction is to be performed on loopers, not factors: {col}"
            assert col in self.frame.columns, f"Trying to reduce a column not in the frame: {col}"
            result = self.reduce_single_index(col)
        return result

    def reduce(self, *args: Union[Looper, str]):
        """usage:
        frame.reduce() to reduce all,
        frame.reduce(col1, col2, ..., col5, col6) to specify first and last columns to reduce
        """
        factor_columns = list(self.factors.keys())

        if len(args) == 0 or (len(args) == 1 and args[0] is Ellipsis):
            return self._reduce([col for col in self.frame.columns[::-1] if col not in factor_columns])
        if Ellipsis not in args:
            result = self._reduce([col.get_name() if isinstance(col, Looper) else col for col in args])
            if result is None:
                logging.warning(
                    "The frame is not fully reduced. Consider using Ellipsis (...) "
                    "to reduce all remaining columns: frame.reduce(col1, col2, ...)."
                )
            return result

        if args.count(Ellipsis) > 1:
            raise ValueError("Only one Ellipsis (...) is allowed in reduce() arguments.")

        ellipsis_index = args.index(Ellipsis)
        columns_before = [col.get_name() if isinstance(col, Looper) else col for col in args[:ellipsis_index]]
        columns_after = [col.get_name() if isinstance(col, Looper) else col for col in args[ellipsis_index + 1 :]]

        frame_columns = [col for col in self.frame.columns if col not in factor_columns]
        ellipsis_columns = [col for col in frame_columns if col not in columns_before + columns_after]
        frame_columns = columns_before + ellipsis_columns + columns_after
        return self._reduce(frame_columns)

    def debug_evaluate_legacy(self):
        for factor_name in list(self.factors.keys()):
            self.merge_factor(factor_name)
        factor_names = list(self.factors.keys())
        return self.frame[factor_names].prod(axis=1).sum()
