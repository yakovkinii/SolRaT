try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import inspect
from collections import OrderedDict
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


_MAX_ROW_ID = 2**62

# Rows wider than this many elements are kept as references to the arrays they came from rather than
# copied into a dense block. A frequency profile is evaluated once per distinct magnetic component
# and used by many rows, so materializing one row-sized copy each costs more than every reduction it
# would speed up: on the Mn I 5432 LTE line, copying the profiles per row tripled the synthesis time.
_MAX_DENSE_ROW_ELEMENTS = 64

# Frames at or below this many rows keep the dict-based join, grouping and uniqueness: the SEE builds
# many small frames, and there a Python dict costs less than factorizing the columns. The two cross
# at roughly this size on the three-column grouping the SEE performs.
_SMALL_FRAME_ROWS = 256

# Expanded index tables, keyed by the sum limits and the base frame they came from. The expansion is
# the same for every synthesis of one atom, so it is built once and copied afterwards. Entries are
# evicted in insertion order.
_INDEX_TABLE_CACHE: "OrderedDict[tuple, _Table]" = OrderedDict()
_INDEX_TABLE_CACHE_SIZE = 8


def _index_table_key(base_frame: pd.DataFrame, sum_limits) -> Union[tuple, None]:
    """
    Cache key of an expanded index table: the sum-limits class and the contents of the base frame.
    Returns ``None`` when the base frame holds values that cannot be hashed, which skips the cache.
    """
    try:
        rows = tuple(map(tuple, base_frame.itertuples(index=False, name=None)))
        return type(sum_limits), tuple(base_frame.columns), rows
    except TypeError:  # pragma: no cover - unhashable cell values
        return None


def clear_index_table_cache() -> None:
    """Drop the cached index tables (they hold one expanded table per atom and sum-limits class)."""
    _INDEX_TABLE_CACHE.clear()


def _row_elements(array: np.ndarray) -> int:
    """Number of elements each row of a dense column carries."""
    return int(np.prod(array.shape[1:], dtype=np.int64)) if array.ndim > 1 else 1


def _as_object_column(values) -> np.ndarray:
    """Object column holding one entry per row, without copying the entries."""
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _as_column(values: List) -> np.ndarray:
    """
    Build a column array from per-row values. Rows may be array-valued (e.g. the T^K_Q or profile
    factors, which carry a vector over Stokes/frequency); those are held in an object array, so rows
    that were joined from the same source array keep sharing it. Purely scalar rows keep their
    natural numeric dtype, which is what lets the reductions run as whole-array numpy operations.
    """
    values = list(values)
    if any(isinstance(value, np.ndarray) for value in values):
        return _as_object_column(values)
    return np.asarray(values)


def _factorize(values: np.ndarray):
    """
    Integer code per element (first-seen order) and the number of distinct elements. Works on the
    numeric index columns and on the object columns holding string ids or J-constraint tuples.
    """
    codes, uniques = pd.factorize(values, sort=False)
    return codes.astype(np.int64, copy=False), max(len(uniques), 1)


def _row_ids(columns: List[np.ndarray]) -> np.ndarray:
    """
    One integer id per row, identifying its combination of values over ``columns``. Each column is
    factorized to small integers and the codes are packed in mixed radix, so that comparing, sorting
    and grouping rows becomes integer work instead of hashing per-row tuples. The packed id is
    compacted whenever it would outgrow the int64 range.
    """
    ids = np.zeros(len(columns[0]), dtype=np.int64)
    cardinality = 1
    for column in columns:
        codes, n_unique = _factorize(column)
        if cardinality * n_unique >= _MAX_ROW_ID:
            ids, cardinality = _factorize(ids)
        ids = ids * n_unique + codes
        cardinality *= n_unique
    return ids


def _concatenate_columns(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Stack two columns of the same logical index for a joint factorization."""
    if first.dtype == second.dtype:
        return np.concatenate([first, second])
    return np.concatenate([np.asarray(first, dtype=object), np.asarray(second, dtype=object)])


def _multiply_columns(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Row-wise product of two columns. Dense blocks are broadcast against each other within the row,
    which reproduces the per-row numpy broadcasting of the object-array representation (a Stokes
    column times a frequency profile gives a Stokes-by-frequency block).
    """
    if first.dtype == object or second.dtype == object:
        # Mixed representation: the dense side becomes one array per row, so the multiplication stays
        # the per-row broadcast the object representation performs.
        if first.dtype != object and first.ndim > 1:
            first = _as_object_column(list(first))
        if second.dtype != object and second.ndim > 1:
            second = _as_object_column(list(second))
        return first * second
    row_ndim = max(first.ndim, second.ndim) - 1
    first = first.reshape((len(first),) + (1,) * (row_ndim - first.ndim + 1) + first.shape[1:])
    second = second.reshape((len(second),) + (1,) * (row_ndim - second.ndim + 1) + second.shape[1:])
    return first * second


def _group_sum(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Sum ``values`` over rows sharing a group code. Dense blocks are summed with one sorted
    ``add.reduceat`` pass; ragged object columns keep the row-by-row accumulation.
    """
    if values.dtype == object:
        accumulator: List = [None] * n_groups
        for code, value in zip(codes, values):
            accumulator[code] = value if accumulator[code] is None else accumulator[code] + value
        return _as_column(accumulator)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.concatenate(([True], sorted_codes[1:] != sorted_codes[:-1])))
    return np.add.reduceat(values[order], boundaries, axis=0)


class _Table:
    r"""
    Minimal columnar table backing the :class:`Frame` engine: an ordered dict of equal-length numpy
    arrays plus a row count (kept explicitly so the empty-0-column seed used for cross joins is
    representable). It replaces the internal pandas ``DataFrame`` so the many small per-cell frame
    operations in the NLTE loop avoid pandas' fixed per-call overhead.

    Index columns keep their natural dtype (float for J/K/Q/M, object for string ids). Factor and
    coefficient columns are dense blocks of shape ``(n_rows, *row_shape)`` when every row carries the
    same shape (a scalar, a Stokes vector, a frequency profile), and object arrays only when the rows
    are ragged. Joins, group-sums and uniqueness pack each row's index values into one integer id
    (:func:`_row_ids`) and then work on those integers with sorting and searching, which keeps them
    whole-array numpy operations and semantically equivalent to the pandas inner-join / groupby-sum /
    drop-duplicates they replace.
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
        data = {}
        for name, array in self.columns.items():
            if array.ndim > 1:  # dense block: hand pandas one array per row
                column = np.empty(len(array), dtype=object)
                for i in range(len(array)):
                    column[i] = array[i]
                data[name] = column
            else:
                data[name] = array
        return pd.DataFrame(data)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "_Table":
        return _Table({column: df[column].to_numpy() for column in df.columns}, n_rows=len(df))

    def unique(self, columns: List[str]) -> "_Table":
        """Distinct rows over ``columns`` (pandas ``drop_duplicates``), preserving first-seen order."""
        if len(columns) == 0:
            return _Table(n_rows=1)
        if self._n_rows == 0:
            return _Table({column: self.columns[column][:0] for column in columns}, n_rows=0)
        if self._n_rows <= _SMALL_FRAME_ROWS:  # dict lookup beats factorizing on the tiny SEE frames
            keys = list(zip(*[self.columns[column] for column in columns]))
            seen, order = set(), []
            for row, key in enumerate(keys):
                if key not in seen:
                    seen.add(key)
                    order.append(row)
            return _Table({column: self.columns[column][order] for column in columns}, n_rows=len(order))
        ids = _row_ids([self.columns[column] for column in columns])
        _, first_occurrence = np.unique(ids, return_index=True)
        order = np.sort(first_occurrence)
        return _Table({column: self.columns[column][order] for column in columns}, n_rows=len(order))

    def add_joined_column(self, name: str, dependency_columns: List[str], source: "_Table") -> None:
        """Add ``name`` by looking each row's ``dependency_columns`` values up in ``source`` (inner join)."""
        source_values = source.columns[name]
        if len(dependency_columns) == 0:
            repeated = np.repeat(source_values[:1], self._n_rows, axis=0)
            self.columns[name] = repeated
            return
        if self._n_rows == 0:
            self.columns[name] = source_values[:0]
            return
        if max(self._n_rows, source.n_rows) <= _SMALL_FRAME_ROWS:
            source_keys = list(zip(*[source.columns[column] for column in dependency_columns]))
            row_of_key = {key: row for row, key in enumerate(source_keys)}
            row_keys = zip(*[self.columns[column] for column in dependency_columns])
            self.columns[name] = source_values[[row_of_key[key] for key in row_keys]]
            return
        # One joint factorization over both tables, so equal values get equal ids on both sides.
        joint_ids = _row_ids(
            [_concatenate_columns(source.columns[column], self.columns[column]) for column in dependency_columns]
        )
        source_ids, row_ids = joint_ids[: source.n_rows], joint_ids[source.n_rows :]  # noqa: E203
        order = np.argsort(source_ids, kind="stable")
        position = np.searchsorted(source_ids[order], row_ids)
        assert position.max(initial=0) < len(order) and np.array_equal(
            source_ids[order][np.clip(position, 0, len(order) - 1)], row_ids
        ), f"Inner join on {dependency_columns} left rows without a match in the source frame."
        row_of_source = order[position]
        if _row_elements(source_values) > _MAX_DENSE_ROW_ELEMENTS:
            # Wide rows (a frequency profile per row): reference the source arrays instead of copying
            # one block per row.
            self.columns[name] = _as_object_column([source_values[i] for i in row_of_source])
        else:
            self.columns[name] = source_values[row_of_source]

    def groupby_sum(self, group_columns: List[str], value_column: str) -> "_Table":
        """Group by ``group_columns`` and sum ``value_column`` (pandas ``groupby(...).sum()``)."""
        if self._n_rows == 0:
            columns = {column: self.columns[column][:0] for column in group_columns}
            columns[value_column] = self.columns[value_column][:0]
            return _Table(columns, n_rows=0)
        if self._n_rows <= _SMALL_FRAME_ROWS:
            keys = list(zip(*[self.columns[column] for column in group_columns]))
            code_of_key: Dict = {}
            codes = np.empty(self._n_rows, dtype=np.int64)
            for row, key in enumerate(keys):
                codes[row] = code_of_key.setdefault(key, len(code_of_key))
            n_groups = len(code_of_key)
        else:
            codes, n_groups = _factorize(_row_ids([self.columns[column] for column in group_columns]))
        _, first_occurrence = np.unique(codes, return_index=True)
        new_columns = {column: self.columns[column][first_occurrence] for column in group_columns}
        new_columns[value_column] = _group_sum(self.columns[value_column], codes, n_groups)
        return _Table(new_columns, n_rows=n_groups)

    def sum_column(self, name: str):
        """Sum a whole column (row entries may be arrays); returns the accumulated value."""
        values = self.columns[name]
        if values.dtype != object:
            return values.sum(axis=0)
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
        r"""
        Build the frame whose rows are the admissible index combinations of ``sum_limits``.

        The expansion depends only on the sum limits and on the values in ``base_frame``: it is the
        atom's index structure, with no reference to the field, the geometry, or the frequency grid.
        It is therefore cached and copied, so that repeated syntheses of the same atom pay for it once.
        """
        key = _index_table_key(base_frame, sum_limits)
        cached = _INDEX_TABLE_CACHE.get(key)
        if cached is not None:
            frame = Frame.__new__(Frame)
            frame._table = cached.copy()
            frame.factors = {}
            frame._n_factors = 0
            return frame

        frame = Frame(base_frame=base_frame, **sum_limits.get_indexes())
        if key is not None:
            if len(_INDEX_TABLE_CACHE) >= _INDEX_TABLE_CACHE_SIZE:
                _INDEX_TABLE_CACHE.pop(next(iter(_INDEX_TABLE_CACHE)))
            _INDEX_TABLE_CACHE[key] = frame._table.copy()
        return frame

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

    @log_method
    def merge_all_factors(self) -> Self:
        """
        Evaluate every registered factor and combine them into one merged factor, without reducing any loop column.
        Use this to precompute the part of a frame whose factors never change, then ``copy()`` it for each use.
        """
        for factor_name in list(self.factors.keys()):
            if not self.factors[factor_name].merged:
                self.merge_factor(factor_name)
        self.combine_all_merged_factors()
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
            sub_frame.columns[factor_name] = raw[:, 0] if raw.shape[1] == 1 else raw
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
            product = _multiply_columns(product, self._table.columns[factor_name])
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
                raise ValueError("Trying to return a partially reduced result")  # pragma: no cover
            return result

        if Ellipsis not in args:
            return self._reduce([col.get_name() if isinstance(col, Looper) else col for col in args])

        if args.count(Ellipsis) > 1:
            raise ValueError("Only one Ellipsis (...) is allowed in reduce() arguments.")  # pragma: no cover

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
