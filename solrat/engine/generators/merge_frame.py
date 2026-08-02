try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import inspect
from typing import Callable, Dict, Generic, List, TypeVar, Union

import numpy as np
import pandas as pd

from solrat.engine.functions.decorators import log_method
from solrat.engine.generators.compiled_operator import CompiledOperator
from solrat.engine.generators.merge_loopers import DummyOrAlreadyMerged, Looper


def merge(df1, df2, on=None):
    """
    Merge helper function with overwritten default behavior. Used only for the one-time looper
    construction (:meth:`Frame.__init__`); the per-call operations use the numpy ``_Table`` backend.
    """
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    if len(on) == 0:
        return df1.merge(df2, how="cross")
    else:
        return df1.merge(df2, on=on, how="inner")


def _as_column(values: List) -> np.ndarray:
    """
    Build a 1-D column array from per-row values. Rows may be array-valued (e.g. the T^K_Q or profile
    factors, which carry a vector over Stokes/frequency); those are held in an object array so numpy
    does not try to stack them into a 2-D block. Purely scalar rows keep their natural numeric dtype.
    """
    values = list(values)
    if any(isinstance(v, np.ndarray) for v in values):
        out = np.empty(len(values), dtype=object)
        for i, value in enumerate(values):
            out[i] = value
        return out
    return np.asarray(values)


def _rebuild_column(values: List, dtype) -> np.ndarray:
    """
    Rebuild a 1-D column from per-row ``values`` (deduplicated / grouped index keys), preserving its
    dtype. Object columns may hold tuples or arrays (e.g. J-constraint sets); assign those element by
    element so numpy keeps a 1-D object array instead of stacking them into a 2-D block, which would
    make the rows unhashable. Scalar columns keep their numeric/string dtype.
    """
    values = list(values)
    if dtype == object:
        out = np.empty(len(values), dtype=object)
        for i, value in enumerate(values):
            out[i] = value
        return out
    return np.asarray(values, dtype=dtype)


class _Table:
    r"""
    Minimal columnar table backing the :class:`Frame` engine: an ordered dict of equal-length numpy
    arrays plus a row count (kept explicitly so the empty-0-column seed used for cross joins is
    representable). It replaces the internal pandas ``DataFrame`` so the many small per-cell frame
    operations in the NLTE loop avoid pandas' fixed per-call overhead.

    Index columns keep their natural dtype (float for J/K/Q/M, object for string ids); factor and
    coefficient columns are object arrays whose entries may themselves be arrays. Joins, group-sums
    and uniqueness are done with plain Python dicts over row-key tuples -- fast on the small frames of
    the hot path, and semantically equivalent to the pandas inner-join / groupby-sum / drop-duplicates
    they replace.
    """

    def __init__(self, columns: Union[Dict[str, np.ndarray], None] = None, n_rows: Union[int, None] = None):
        self.columns: Dict[str, np.ndarray] = dict(columns) if columns is not None else {}
        if n_rows is not None:
            self._n_rows = int(n_rows)
        elif self.columns:
            self._n_rows = len(next(iter(self.columns.values())))
        else:
            self._n_rows = 0

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def names(self) -> List[str]:
        return list(self.columns.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def copy(self) -> "_Table":
        return _Table({name: array.copy() for name, array in self.columns.items()}, n_rows=self._n_rows)

    def drop(self, name: str) -> None:
        del self.columns[name]

    def rename(self, old: str, new: str) -> None:
        self.columns[new] = self.columns.pop(old)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.columns:
            return pd.DataFrame(index=range(self._n_rows), columns=[])
        return pd.DataFrame(self.columns)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "_Table":
        return _Table({column: df[column].to_numpy() for column in df.columns}, n_rows=len(df))

    def unique(self, columns: List[str]) -> "_Table":
        """Distinct rows over ``columns`` (pandas ``drop_duplicates``), preserving first-seen order."""
        if len(columns) == 0:
            return _Table(n_rows=1)
        keys = list(zip(*[self.columns[column] for column in columns]))
        seen = set()
        order = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                order.append(key)
        new_columns = {
            column: _rebuild_column([key[j] for key in order], self.columns[column].dtype)
            for j, column in enumerate(columns)
        }
        return _Table(new_columns, n_rows=len(order))

    def add_joined_column(self, name: str, dependency_columns: List[str], source: "_Table") -> None:
        """Add ``name`` by looking each row's ``dependency_columns`` tuple up in ``source`` (inner join)."""
        source_values = source.columns[name]
        if len(dependency_columns) == 0:
            self.columns[name] = _as_column([source_values[0]] * self._n_rows)
            return
        source_keys = list(zip(*[source.columns[column] for column in dependency_columns]))
        value_by_key = {key: value for key, value in zip(source_keys, source_values)}
        row_keys = list(zip(*[self.columns[column] for column in dependency_columns]))
        self.columns[name] = _as_column([value_by_key[key] for key in row_keys])

    def groupby_sum(self, group_columns: List[str], value_column: str) -> "_Table":
        """Group by ``group_columns`` and sum ``value_column`` (pandas ``groupby(...).sum()``)."""
        keys = list(zip(*[self.columns[column] for column in group_columns]))
        values = self.columns[value_column]
        accumulator: Dict = {}
        order = []
        for key, value in zip(keys, values):
            if key in accumulator:
                accumulator[key] = accumulator[key] + value
            else:
                accumulator[key] = value
                order.append(key)
        new_columns = {
            column: _rebuild_column([key[j] for key in order], self.columns[column].dtype)
            for j, column in enumerate(group_columns)
        }
        new_columns[value_column] = _as_column([accumulator[key] for key in order])
        return _Table(new_columns, n_rows=len(order))

    def sum_column(self, name: str):
        """Sum a whole column (row entries may be arrays); returns the accumulated value."""
        values = self.columns[name]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total


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

    The table is held in a numpy-backed ``_Table`` (:attr:`_table`); the ``frame`` property exposes it
    as a pandas ``DataFrame`` for backward compatibility (reading, copying, or assigning ``.frame``).
    """

    @staticmethod
    def from_sum_limits(base_frame: pd.DataFrame, sum_limits: SumLimitsT) -> "Frame":
        looper_dict = sum_limits.get_indexes()
        return Frame(base_frame=base_frame, **looper_dict)

    def __init__(self, base_frame: Union[pd.DataFrame, None] = None, **kwargs: Looper):
        # The one-time looper construction is done with pandas (loopers operate on DataFrames), then
        # converted to the numpy ``_Table`` backend used by all per-call operations.
        if base_frame is not None:
            df: pd.DataFrame = base_frame.copy()
        else:
            df = pd.DataFrame(index=[0], columns=[])

        for looper_name, looper in kwargs.items():
            looper.set_name(looper_name)
            if isinstance(looper, DummyOrAlreadyMerged):
                continue
            dependent_cols = list(looper.get_directly_dependent_columns())
            if len(dependent_cols) == 0:
                sub_frame = pd.DataFrame(index=[0], columns=[])
            else:
                sub_frame = df[dependent_cols].drop_duplicates().reset_index(drop=True)
            sub_frame_filled = looper.fill_frame(sub_frame)
            assert not sub_frame_filled[looper_name].isna().any()
            df = merge(df, sub_frame_filled)

        self._table: _Table = _Table.from_dataframe(df)
        self.factors: Dict[str, FrameFactor] = {}
        self._n_factors = 0  # for naming only

    @property
    def frame(self) -> pd.DataFrame:
        """Pandas view of the table (backward compatibility for modeler code that reads ``.frame``)."""
        return self._table.to_dataframe()

    @frame.setter
    def frame(self, value: pd.DataFrame) -> None:
        self._table = _Table.from_dataframe(value)

    @log_method
    def copy(self):
        # Bypass __init__: it builds (and _Table-converts) a throwaway empty pandas DataFrame that we
        # immediately overwrite below. copy() is on the per-cell hot path, so that empty-DataFrame
        # construction dominated the runtime; __new__ sets up the instance without it.
        new_frame = Frame.__new__(Frame)
        new_frame._table = self._table.copy()
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

    def construct_sub_frame(self, columns: List[str]) -> _Table:
        """
        This is used to reduce the evaluations of loopers/factors to minimum:
        we get all unique dependencies, evaluate on them, then merge back to the frame.
        """
        if len(columns) == 0:
            return _Table(n_rows=1)
        return self._table.unique(columns)

    @log_method
    def register_multiplication(self, *args: Callable, elementwise: bool = False, **kwargs):
        """
        This just registers the factors. They will be evaluated/merged later on demand.
        """
        for factor_callable in args:
            name = f"factor_{self._n_factors}"
            assert name not in self._table, f"Cannot add {name} as a factor: name already used."
            self.factors[name] = FrameFactor(name, factor_callable, elementwise=elementwise)
            self._n_factors += 1

        for name, factor_callable in kwargs.items():
            assert name not in self._table, f"Cannot add {name} as a factor: name already used."
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

        sub_frame = self.construct_sub_frame(factor.dependencies)
        if factor.elementwise:
            # Do it row-wise, because the factor does not support array inputs.
            values = []
            for i in range(sub_frame.n_rows):
                row_arguments = {name: sub_frame.columns[name][i] for name in factor.dependencies}
                values.append(factor.call(**row_arguments))
            sub_frame.columns[factor_name] = _as_column(values)
        else:
            # Regular logic: evaluate the factor on the whole (deduplicated) dependency stack at once.
            arguments = {name: sub_frame.columns[name].reshape(-1, 1) for name in factor.dependencies}
            raw = np.reshape(np.asarray(factor.call(**arguments)), (sub_frame.n_rows, -1))
            if raw.shape[1] == 1:
                sub_frame.columns[factor_name] = raw[:, 0]
            else:
                sub_frame.columns[factor_name] = _as_column([raw[i] for i in range(sub_frame.n_rows)])
        self._table.add_joined_column(factor_name, factor.dependencies, sub_frame)
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
        product = self._table.columns[factor_names[0]]
        for factor_name in factor_names[1:]:
            product = product * self._table.columns[factor_name]
        self._table.columns[new_factor_name] = product
        dependencies = list(set().union(*[self.factors[name].dependencies for name in factor_names]))
        self.factors[new_factor_name] = FrameFactor(new_factor_name, dependencies=dependencies, merged=True)

        for factor_name in factor_names:
            del self._table.columns[factor_name]
            del self.factors[factor_name]

        return new_factor_name

    def remove_dependency(self, column: str):
        for factor in self.factors.values():
            if column in factor.dependencies:
                assert factor.merged, "Trying to remove a column dependency from unmerged factor"
                factor.dependencies.remove(column)

    def get_other_frame_columns(self, exclude: str) -> List[str]:
        """Get looper columns other than the specified one"""
        return [col for col in self._table.names if col != exclude and col not in self.factors]

    def reduce_single_index(self, column: Union[str, Looper]):
        """
        Reduction is Looper-wise (this way it clearly follows the logic of 'summation' operation)
        """
        if isinstance(column, Looper):
            column = column.get_name()

        dependent_factors = self.get_dependent_factors(column)

        if len(dependent_factors) == 0:
            self.remove_dependency(column)
            self._table.drop(column)
            return self

        for factor_name in dependent_factors:
            if not self.factors[factor_name].merged:
                self.merge_factor(factor_name)

        factor_name = self.combine_all_merged_factors()
        self.remove_dependency(column)

        group_columns = self.get_other_frame_columns(column)

        if len(group_columns) == 0:
            assert len(self.factors) == 1, f"Reduced all loopers, but some factors remain: {self.factors}"
            return self._table.sum_column(factor_name)

        self._table = self.groupby_sum_keeping_factors(group_columns, factor_name)
        return self

    def groupby_sum_keeping_factors(self, group_columns: List[str], value_column: str) -> _Table:
        """Group-sum the value column over ``group_columns`` (the surviving loopers)."""
        return self._table.groupby_sum(group_columns, value_column)

    def _reduce(self, columns) -> Union[np.ndarray, float, complex, Self]:
        result = None
        for col in columns:
            assert col not in self.factors, f"Reduction is to be performed on loopers, not factors: {col}"
            assert col in self._table, f"Trying to reduce a column not in the frame: {col}"
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
            result = self._reduce([col for col in self._table.names[::-1] if col not in factor_columns])
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

        frame_columns = [col for col in self._table.names if col not in factor_columns]
        ellipsis_columns = [col for col in frame_columns if col not in columns_before + columns_after]
        frame_columns = columns_before + ellipsis_columns + columns_after
        return self._reduce(frame_columns)

    @log_method
    def to_operator(
        self, ordered_multiplicand_keys: List[str], coefficient_col: str = "coefficient"
    ) -> CompiledOperator:
        r"""
        Capture this reduced frame as a :class:`CompiledOperator`: the rows keyed by
        ``ordered_multiplicand_keys`` (the loop columns left unmerged, in the order the operator is
        applied against) and the single ``coefficient_col`` column. The remaining
        ``register_multiplication(value) + reduce`` over those keys is then ``operator.multiply(value)``.
        """
        return CompiledOperator.from_columns(self._table.columns, ordered_multiplicand_keys, coefficient_col)

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
        self._table.rename(combined_name, new_factor_name)
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
        combined_name = self.combine_all_merged_factors()
        return self._table.sum_column(combined_name)
