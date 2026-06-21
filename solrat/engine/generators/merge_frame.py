try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import inspect
from typing import Callable, Dict, Generic, List, TypeVar, Union

import numpy as np
import pandas as pd

from solrat.engine.functions.decorators import log_method
from solrat.engine.generators.merge_loopers import DummyOrAlreadyMerged, Looper


def merge(df1, df2, on=None):
    """
    Merge helper function with overwritten default behavior.
    """
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    if len(on) == 0:
        return df1.merge(df2, how="cross")
    else:
        return df1.merge(df2, on=on, how="inner")


class SumLimits:
    r"""
    Sum limits base class.

    The inheriting classes control the limits for the summation indexes such as :math:`L, J, K, Q,` etc.
    We start from the 'base_frame' which has some
    indexes and quantities already pre-merged, like :math:`L_l, L_u, S`, Einstein coefficients.
    Then we can determine the boundaries of the summation indexes that follow.

    Triangular means from :math:`|a-b|` to :math:`a + b` (both ends included)

    FromTo means from :math:`a` to :math:`b` (both ends included)

    Intersection means including only shared values of 2 or more sets of values.

    For further information inspect each Looper individually.
    """

    @classmethod
    def get_indexes(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__") and not callable(v)}

    def __init__(self):
        indexes = self.get_indexes()
        for k, v in indexes.items():
            v.name = None


SumLimitsT = TypeVar("SumLimitsT", bound=SumLimits)


class FrameFactor:
    """
    Single multiplicand in the frame
    """

    def __init__(
        self,
        name: str,
        factor: Union[Callable, None] = None,
        dependencies: Union[List[str], None] = None,
        merged: bool = False,
        elementwise: bool = False,
    ):
        self.name: str = name
        self.call: Union[Callable, None] = factor
        if dependencies is not None:
            self.dependencies: List[str] = dependencies
        else:
            assert factor is not None
            self.dependencies: List[str] = [p.name for p in inspect.signature(factor).parameters.values()]
        self.merged: bool = merged
        self.elementwise: bool = elementwise
        # logging.log(VERBOSE, f"Created: {self}")

    def __repr__(self):
        return (
            f"FrameFactor {self.name}. Dependencies: {self.dependencies}. Merged: {self.merged}. "
            f"Elementwise: {self.elementwise}"
        )

    @log_method
    def copy(self):
        return FrameFactor(
            name=self.name,
            factor=self.call,
            dependencies=self.dependencies.copy(),
            merged=self.merged,
            elementwise=self.elementwise,
        )


class Frame(Generic[SumLimitsT]):
    r"""
    Frame engine for performing multiplication and summation.

    Loopers are merged immediately to the base frame.
    Factors are stored and evaluated+merged when needed.
    """

    @staticmethod
    def from_sum_limits(base_frame: pd.DataFrame, sum_limits: SumLimitsT) -> "Frame":
        looper_dict = sum_limits.get_indexes()
        return Frame(base_frame=base_frame, **looper_dict)

    def __init__(self, base_frame: Union[pd.DataFrame, None] = None, **kwargs: Looper):
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
            # logging.log(VERBOSE, f"Merged {looper_name}, frame shape = {self.frame.shape}")

        self.factors: Dict[str, FrameFactor] = {}
        self._n_factors = 0  # for naming only
        # logging.log(VERBOSE, f"Frame shape after initialization: {self.frame.shape}")

    @log_method
    def copy(self):
        new_frame = Frame()
        new_frame.frame = self.frame.copy()
        new_frame.factors = {k: factor.copy() for k, factor in self.factors.items()}
        new_frame._n_factors = self._n_factors
        return new_frame

    def __repr__(self):  # pragma: no cover
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
        for fn, factor in self.factors.items():
            result += f"{fn}: {factor}\n"

        return result

    def construct_sub_frame(self, columns: List[str]) -> pd.DataFrame:
        """
        This is used to reduce the evaluations of loopers/factors to minimum:
        we get all unique dependencies, evaluate on them, then merge back to the frame.
        """
        if len(columns) == 0:
            return pd.DataFrame(index=[0], columns=[])
        return self.frame[columns].drop_duplicates().reset_index(drop=True)

    @log_method
    def register_multiplication(self, *args: Callable, elementwise: bool = False, **kwargs):
        """
        This just registers the factors. They will be evaluated/merged later on demand.
        """
        for factor_callable in args:
            name = f"factor_{self._n_factors}"
            assert name not in self.frame.columns, f"Cannot add {name} as a factor: name already used."
            self.factors[name] = FrameFactor(name, factor_callable, elementwise=elementwise)
            self._n_factors += 1

        for name, factor_callable in kwargs.items():
            assert name not in self.frame.columns, f"Cannot add {name} as a factor: name already used."
            self.factors[name] = FrameFactor(name, factor_callable, elementwise=elementwise)
            self._n_factors += 1

        return self

    def get_dependent_factors(self, column: str) -> List[str]:
        return [name for name, factor in self.factors.items() if column in factor.dependencies]

    def merge_factor(self, factor_name: str):
        """
        Construct factor frame, evaluate, and merge it to the main frame
        """
        factor = self.factors[factor_name]
        # logging.log(VERBOSE, f"Merging factor: {factor}")

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

    def combine_all_merged_factors(self) -> str:
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
        self.factors[new_factor_name] = FrameFactor(new_factor_name, dependencies=dependencies, merged=True)

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

        # logging.log(VERBOSE, "====")
        # logging.log(VERBOSE, f"Reducing column {column}:")
        dependent_factors = self.get_dependent_factors(column)
        # logging.log(VERBOSE, f"Dependent factors: {dependent_factors}")
        # logging.log(VERBOSE, f"Dependent factors details: {[self.factors[df] for df in dependent_factors]}")

        if len(dependent_factors) == 0:
            # logging.log(VERBOSE, f"No dependent factors for column {column}, dropping it directly.")
            self.remove_dependency(column)
            self.frame = self.frame.drop(columns=column)
            return self

        for factor_name in dependent_factors:
            # logging.log(VERBOSE, f"Ensuring factor {factor_name} is merged for reduction.")
            if not self.factors[factor_name].merged:
                # logging.log(VERBOSE, f"    Merging factor {factor_name} now.")
                self.merge_factor(factor_name)

        factor_name = self.combine_all_merged_factors()
        # logging.log(VERBOSE, f"Combined dependent factors into {factor_name} for reduction.")
        self.remove_dependency(column)

        group_columns = self.get_other_frame_columns(column)
        # logging.log(VERBOSE, f"  Grouping by columns: {group_columns} to reduce {column}.")

        if len(group_columns) == 0:
            # logging.log(VERBOSE, "  Reduced the last looper!")
            assert len(self.factors) == 1, f"Reduced all loopers, but some factors remain: {self.factors}"
            # self.frame = self.frame.drop(columns=column)
            # logging.log(VERBOSE, f"  No grouping columns left, returning sum of {factor_name}.")
            # logging.log(VERBOSE, "Calculating the sum over the last looper and returning the result")
            return self.frame[factor_name].sum()

        self.frame = self.frame.groupby(group_columns)[factor_name].sum().reset_index()
        # logging.log(VERBOSE, f"  Reduced frame shape: {self.frame.shape}")
        return self

    def _reduce(self, columns) -> Union[np.ndarray, float, complex, Self]:
        result = None
        for col in columns:
            assert col not in self.factors, f"Reduction is to be performed on loopers, not factors: {col}"
            assert col in self.frame.columns, f"Trying to reduce a column not in the frame: {col}"
            result = self.reduce_single_index(col)
        return result

    @log_method
    def reduce(self, *args: Union[Looper, str]) -> Union[np.ndarray, float, complex, Self]:
        r"""usage:
        frame.reduce() to reduce all,
        frame.reduce(col1, col2, ..., col5, col6) to specify first and last columns to reduce
        """
        factor_columns = list(self.factors.keys())

        if len(args) == 0 or (len(args) == 1 and args[0] is Ellipsis):
            result = self._reduce([col for col in self.frame.columns[::-1] if col not in factor_columns])
            if result is None:
                raise ValueError("Trying to return a partially reduced result")
            return result

        if Ellipsis not in args:
            return self._reduce([col.get_name() if isinstance(col, Looper) else col for col in args])

        if args.count(Ellipsis) > 1:
            raise ValueError("Only one Ellipsis (...) is allowed in reduce() arguments.")

        ellipsis_index = args.index(Ellipsis)
        columns_before = [col.get_name() if isinstance(col, Looper) else col for col in args[:ellipsis_index]]
        columns_after = [
            col.get_name() if isinstance(col, Looper) else col for col in args[ellipsis_index + 1 :]  # noqa: E203
        ]

        frame_columns = [col for col in self.frame.columns if col not in factor_columns]
        ellipsis_columns = [col for col in frame_columns if col not in columns_before + columns_after]
        frame_columns = columns_before + ellipsis_columns + columns_after
        return self._reduce(frame_columns)

    @log_method
    def to_coefficient(self) -> Self:
        """
        Merge all registered factors into a single 'coefficient' column without reducing any loop columns.
        Use this when you want the coefficient but still need all loop columns for index lookup.
        """
        for factor_name in list(self.factors.keys()):
            if not self.factors[factor_name].merged:
                self.merge_factor(factor_name)
        combined_name = self.combine_all_merged_factors()

        new_factor_name = "coefficient"
        self.frame = self.frame.rename(columns={combined_name: new_factor_name})
        dependencies = self.factors[combined_name].dependencies
        self.factors[new_factor_name] = FrameFactor(new_factor_name, dependencies=dependencies, merged=True)
        del self.factors[combined_name]

        return self

    @log_method
    def reduce_partially(self, *args: Union[Looper, str]) -> Self:
        """Reduce the given loop columns (groupby-sum) then merge all factors to 'coefficient'."""
        self.reduce(*args)
        return self.to_coefficient()

    @log_method
    def debug_reduce_legacy(self):  # pragma: no cover
        for factor_name in list(self.factors.keys()):
            self.merge_factor(factor_name)
        factor_names = list(self.factors.keys())
        return self.frame[factor_names].prod(axis=1).sum()
