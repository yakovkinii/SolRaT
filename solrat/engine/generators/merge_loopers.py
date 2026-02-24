"""
TODO
TODO  This file needs improved documentation.
TODO
"""

import logging
from abc import abstractmethod
from functools import reduce
from typing import Set, Union

import numpy as np
import pandas as pd

MAX_ID = 0


def vector(arr):
    return pd.Series([row for row in arr])


def get_unique_name() -> str:
    global MAX_ID
    name = f"__looper_unique_{MAX_ID}__"
    MAX_ID += 1
    return name


class Looper:
    def __init__(self):
        self.name = None
        self.is_name_user_set = None

    def set_name(self, name: str):
        assert self.name is None
        self.name = name
        self.is_name_user_set = True

    def get_name(self) -> str:
        if self.name is None:
            self.name = get_unique_name()
            self.is_name_user_set = False
        return self.name

    def add_to_dependencies(self, dependencies: Set[str]):
        msg = "Looper name should be set or determined before adding to dependencies. "
        msg += "Check that all loopers are passed to Frame."
        assert self.is_name_user_set is not None, msg
        if self.is_name_user_set:
            dependencies.add(self.get_name())

    def is_name_set(self) -> bool:  # Todo remove
        return self.name is not None

    def get_directly_dependent_columns(self) -> Set[str]:
        return set()

    def get_dependent_columns(self) -> Set[str]:
        return set()

    @abstractmethod
    def fill_frame(self, frame: pd.DataFrame, explode: bool = True) -> pd.DataFrame:
        pass

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        assert isinstance(other, Looper)
        return Sum(self, other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        assert isinstance(other, Looper)
        return Difference(self, other)


class DummyOrAlreadyMerged(Looper):
    def __init__(self, dependency: Union[Looper, None] = None):
        super().__init__()
        self.dependency: Union[Looper, None] = dependency

    def add_to_dependencies(self, dependencies: Set[str]):
        if self.is_name_user_set:
            dependencies.add(self.get_name())
            return
        if self.dependency is not None:
            self.dependency.add_to_dependencies(dependencies)

    def get_directly_dependent_columns(self) -> Set[str]:
        cols = set()
        if self.dependency is not None:
            self.dependency.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> Set[str]:
        cols = set()
        if self.dependency is not None:
            self.dependency.add_to_dependencies(cols)
            cols.update(self.dependency.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        raise ValueError()


class Value(Looper):
    def __init__(self, value: Union[int, float, str]):
        super().__init__()
        self.value = value

    def add_to_dependencies(self, dependencies: Set[str]):
        pass

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        frame[name] = self.value
        if isinstance(self.value, (int, float)):
            frame = frame.astype({name: float})
        elif isinstance(self.value, str):
            frame = frame.astype({name: str})
        return frame


def wrap_in_value_if_needed(x: Union[Looper, int, float, str]) -> Looper:
    if isinstance(x, Looper):
        return x
    if isinstance(x, (int, float)):
        return Value(x)
    if isinstance(x, str):
        logging.warning(f"Wrapping a str value {x} in Value looper.")
        return Value(x)
    raise ValueError(f"Cannot wrap type {type(x)} in Value looper.")


class FromTo(Looper):
    def __init__(self, start: Union[Looper, int, float], end: Union[Looper, int, float]):
        super().__init__()
        self.start: Looper = wrap_in_value_if_needed(start)
        self.end: Looper = wrap_in_value_if_needed(end)

    def add_to_dependencies(self, dependencies: Set[str]):
        if self.is_name_user_set:
            dependencies.add(self.get_name())
            return
        self.start.add_to_dependencies(dependencies)
        self.end.add_to_dependencies(dependencies)

    def get_directly_dependent_columns(self) -> Set[str]:
        cols = set()
        self.start.add_to_dependencies(cols)
        self.end.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> Set[str]:
        cols = set()
        self.start.add_to_dependencies(cols)
        cols.update(self.start.get_dependent_columns())
        self.end.add_to_dependencies(cols)
        cols.update(self.end.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        if not self.start.is_name_user_set:
            frame = self.start.fill_frame(frame)
        if not self.end.is_name_user_set:
            frame = self.end.fill_frame(frame)
        start_name = self.start.get_name()
        end_name = self.end.get_name()
        frame[name] = frame.apply(lambda row: list(np.arange(row[start_name], row[end_name] + 1)), axis=1)
        if not self.start.is_name_user_set:
            frame = frame.drop(columns=[start_name])
        if not self.end.is_name_user_set:
            frame = frame.drop(columns=[end_name])
        if explode:
            frame = frame.explode(name)
            frame = frame.astype({name: float})
        return frame


class Projection(Looper):
    def __init__(self, vector: Union[Looper, int, float]):
        super().__init__()
        self.vector: Looper = wrap_in_value_if_needed(vector)

    def add_to_dependencies(self, dependencies: Set[str]):
        if self.is_name_user_set:
            dependencies.add(self.get_name())
            return
        self.vector.add_to_dependencies(dependencies)

    def get_directly_dependent_columns(self) -> Set[str]:
        cols = set()
        self.vector.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> Set[str]:
        cols = set()
        self.vector.add_to_dependencies(cols)
        cols.update(self.vector.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        if not self.vector.is_name_user_set:
            frame = self.vector.fill_frame(frame)
        vector_name = self.vector.get_name()
        frame[name] = frame.apply(lambda row: list(np.arange(-row[vector_name], row[vector_name] + 1)), axis=1)
        if not self.vector.is_name_user_set:
            frame = frame.drop(columns=[vector_name])
        if explode:
            frame = frame.explode(name)
            frame = frame.astype({name: float})
        return frame


class Triangular(Looper):
    def __init__(self, vector1: Union[Looper, int, float], vector2: Union[Looper, int, float]):
        super().__init__()
        self.vector1: Looper = wrap_in_value_if_needed(vector1)
        self.vector2: Looper = wrap_in_value_if_needed(vector2)

    def add_to_dependencies(self, dependencies: Set[str]):
        if self.is_name_user_set:
            dependencies.add(self.get_name())
            return
        self.vector1.add_to_dependencies(dependencies)
        self.vector2.add_to_dependencies(dependencies)

    def get_directly_dependent_columns(self) -> Set[str]:
        cols = set()
        self.vector1.add_to_dependencies(cols)
        self.vector2.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> Set[str]:
        cols = set()
        self.vector1.add_to_dependencies(cols)
        cols.update(self.vector1.get_dependent_columns())
        self.vector2.add_to_dependencies(cols)
        cols.update(self.vector2.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        if not self.vector1.is_name_user_set:
            frame = self.vector1.fill_frame(frame)
        if not self.vector2.is_name_user_set:
            frame = self.vector2.fill_frame(frame)
        vector1_name = self.vector1.get_name()
        vector2_name = self.vector2.get_name()
        frame[name] = frame.apply(
            lambda row: list(
                np.arange(np.abs(row[vector1_name] - row[vector2_name]), row[vector1_name] + row[vector2_name] + 1)
            ),
            axis=1,
        )
        if not self.vector1.is_name_user_set:
            frame = frame.drop(columns=[vector1_name])
        if not self.vector2.is_name_user_set:
            frame = frame.drop(columns=[vector2_name])
        if explode:
            frame = frame.explode(name)
            frame = frame.astype({name: float})
        return frame


class Sum(Looper):
    def __init__(self, left: Looper, right: Looper):
        super().__init__()
        self.left = wrap_in_value_if_needed(left)
        self.right = wrap_in_value_if_needed(right)
        self.names_set = False

    def add_to_dependencies(self, dependencies: Set[str]):
        if self.is_name_user_set:
            dependencies.add(self.get_name())
            return
        self.left.add_to_dependencies(dependencies)
        self.right.add_to_dependencies(dependencies)

    def get_directly_dependent_columns(self) -> Set[str]:
        cols = set()
        self.left.add_to_dependencies(cols)
        self.right.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> Set[str]:
        cols = set()
        self.left.add_to_dependencies(cols)
        cols.update(self.left.get_dependent_columns())
        self.right.add_to_dependencies(cols)
        cols.update(self.right.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        if not self.left.is_name_user_set:
            frame = self.left.fill_frame(frame)
        if not self.right.is_name_user_set:
            frame = self.right.fill_frame(frame)
        frame[name] = frame[self.left.get_name()] + frame[self.right.get_name()]

        if not self.left.is_name_user_set:
            frame = frame.drop(columns=[self.left.get_name()])
        if not self.right.is_name_user_set:
            frame = frame.drop(columns=[self.right.get_name()])

        return frame


class Difference(Looper):
    def __init__(self, left: Looper, right: Looper):
        super().__init__()
        self.left = wrap_in_value_if_needed(left)
        self.right = wrap_in_value_if_needed(right)

    def add_to_dependencies(self, dependencies: Set[str]):
        if self.is_name_user_set:
            dependencies.add(self.get_name())
            return
        self.left.add_to_dependencies(dependencies)
        self.right.add_to_dependencies(dependencies)

    def get_directly_dependent_columns(self) -> Set[str]:
        cols = set()
        self.left.add_to_dependencies(cols)
        self.right.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> Set[str]:
        cols = set()
        self.left.add_to_dependencies(cols)
        cols.update(self.left.get_dependent_columns())
        self.right.add_to_dependencies(cols)
        cols.update(self.right.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        if not self.left.is_name_user_set:
            frame = self.left.fill_frame(frame)
        if not self.right.is_name_user_set:
            frame = self.right.fill_frame(frame)
        frame[name] = frame[self.left.get_name()] - frame[self.right.get_name()]

        if not self.left.is_name_user_set:
            frame = frame.drop(columns=[self.left.get_name()])
        if not self.right.is_name_user_set:
            frame = frame.drop(columns=[self.right.get_name()])

        return frame


class Intersection(Looper):
    def __init__(self, *args: Looper):
        super().__init__()
        self.loopers = [wrap_in_value_if_needed(arg) for arg in args]
        assert all([not isinstance(looper, Intersection) for looper in self.loopers])

    def get_directly_dependent_columns(self) -> set:
        cols = set()
        for looper in self.loopers:
            looper.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> set:
        cols = set()
        for looper in self.loopers:
            looper.add_to_dependencies(cols)
            cols.update(looper.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        assert explode, "Intersection looper requires explode=True"

        name = self.get_name()
        for looper in self.loopers:
            if not looper.is_name_user_set:
                frame = looper.fill_frame(frame, explode=False)

        cols = [looper.get_name() for looper in self.loopers]
        frame[name] = frame.apply(lambda row: list(reduce(np.intersect1d, [row[col] for col in cols])), axis=1)

        for looper in self.loopers:
            if not looper.is_name_user_set:
                frame = frame.drop(columns=[looper.get_name()])

        frame = frame.explode(name)

        if frame[name].isna().any():
            msg = (
                f"NaN values found in intersection looper {name}. "
                f"Check previous loopers for possible optimization (triangular conditions etc). "
                f"If not applicable - double check the logic."
            )
            logging.warning(msg)
        frame = frame.dropna(subset=[name])
        frame = frame.astype({name: float})

        return frame


class Constraint(DummyOrAlreadyMerged):
    """
    Constrains the values of some variable to a list of values.
    This is meant to be an artificial constraint, not triangular/etc.
    """

    def __init__(self):
        super().__init__()


class ApplyConstraint(Looper):
    def __init__(self, looper: Looper, constraint: Constraint):
        super().__init__()
        # self.loopers = [wrap_in_value_if_needed(arg) for arg in args]
        self.looper = wrap_in_value_if_needed(looper)
        self.constraint = constraint

    def get_directly_dependent_columns(self) -> set:
        cols = set()
        self.looper.add_to_dependencies(cols)
        self.constraint.add_to_dependencies(cols)
        return cols

    def get_dependent_columns(self) -> set:
        cols = set()
        self.looper.add_to_dependencies(cols)
        cols.update(self.looper.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        assert explode, "Intersection looper requires explode=True"
        assert not self.looper.is_name_user_set, "Cannot constrain a named looper"
        frame = self.looper.fill_frame(frame).reset_index(drop=True)

        looper_name = self.looper.get_name()
        constraint_name = self.constraint.get_name()

        mask = [False] * len(frame)
        # Slow but should work:
        for i in range(len(frame)):
            if frame.at[i, looper_name] in frame.at[i, constraint_name] or frame.at[i, constraint_name] is None:
                mask[i] = True

        frame = frame[mask].reset_index(drop=True)
        frame = frame.rename(columns={looper_name: self.get_name()})
        return frame
