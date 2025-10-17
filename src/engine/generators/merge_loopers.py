import logging
from typing import Union
from functools import reduce
import numpy as np
import pandas as pd
import inspect

from abc import ABC, abstractmethod

MAX_ID = 0
def get_unique_name()->str:
    global MAX_ID
    name = f"__looper_unique_{MAX_ID}__"
    MAX_ID += 1
    return name

class Looper:
    def __init__(self):
        self.name = None

    def set_name(self, name:str):
        assert self.name is None
        self.name = name

    def get_name(self)->str:
        assert self.name is not None
        return self.name

    def is_name_set(self)->bool:
        return self.name is not None

    def get_directly_dependent_columns(self)->set:
        return set()

    def get_dependent_columns(self)->set:
        return set()

    @abstractmethod
    def fill_frame(self, frame: pd.DataFrame, explode:bool=True) -> pd.DataFrame:
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



class Dummy(Looper):
    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        raise ValueError()


class Value(Looper):
    def __init__(self, value:Union[int, float, str]):
        super().__init__()
        self.value = value

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        frame[name] = self.value
        if isinstance(self.value, (int, float)):
            frame = frame.astype({name: float})
        elif isinstance(self.value, str):
            frame = frame.astype({name: str})
        return frame

class FromTo(Looper):
    def __init__(self, start:Union[Looper, int, float], end:Union[Looper, int, float]):
        super().__init__()
        self.start:Union[Looper, int, float] = start
        self.end:Union[Looper, int, float] = end

    def get_directly_dependent_columns(self)->set:
        cols = set()
        if isinstance(self.start, Looper):
            cols.add(self.start.get_name())
        if isinstance(self.end, Looper):
            cols.add(self.end.get_name())
        return cols

    def get_dependent_columns(self)->set:
        cols = set()
        if isinstance(self.start, Looper):
            cols.add(self.start.get_name())
            cols.update(self.start.get_dependent_columns())
        if isinstance(self.end, Looper):
            cols.add(self.end.get_name())
            cols.update(self.end.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()

        start_values = frame[self.start.get_name()] if isinstance(self.start, Looper) else self.start
        end_values = frame[self.end.get_name()] if isinstance(self.end, Looper) else self.end
        frame['__start__'] = start_values
        frame['__end__'] = end_values
        frame[name] = frame.apply(lambda row: list(np.arange(row['__start__'], row['__end__'] + 1)), axis=1)
        frame = frame.drop(columns=['__start__', '__end__'])
        if explode:
            frame = frame.explode(name)
            frame = frame.astype({name: float})
        return frame


class Projection(Looper):
    def __init__(self, vector:Union[Looper, int, float]):
        super().__init__()
        self.vector:Union[Looper, int, float] = vector


    def get_directly_dependent_columns(self)->set:
        cols = set()
        if isinstance(self.vector, Looper):
            cols.add(self.vector.get_name())
        return cols

    def get_dependent_columns(self)->set:
        cols = set()
        if isinstance(self.vector, Looper):
            cols.add(self.vector.get_name())
            cols.update(self.vector.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        vector_values = frame[self.vector.get_name()] if isinstance(self.vector, Looper) else self.vector
        frame['__start__'] = -vector_values
        frame['__end__'] = vector_values
        frame[name] = frame.apply(lambda row: list(np.arange(row['__start__'], row['__end__'] + 1)), axis=1)
        frame = frame.drop(columns=['__start__', '__end__'])
        if explode:
            frame = frame.explode(name)
            frame = frame.astype({name: float})
        return frame



class Triangular(Looper):
    def __init__(self, vector1:Union[Looper, int, float], vector2:Union[Looper, int, float]):
        super().__init__()
        self.vector1:Union[Looper, int, float] = vector1
        self.vector2:Union[Looper, int, float] = vector2

    def get_directly_dependent_columns(self)->set:
        cols = set()
        if isinstance(self.vector1, Looper):
            cols.add(self.vector1.get_name())
        if isinstance(self.vector2, Looper):
            cols.add(self.vector2.get_name())
        return cols

    def get_dependent_columns(self)->set:
        cols = set()
        if isinstance(self.vector1, Looper):
            cols.add(self.vector1.get_name())
            cols.update(self.vector1.get_dependent_columns())
        if isinstance(self.vector2, Looper):
            cols.add(self.vector2.get_name())
            cols.update(self.vector2.get_dependent_columns())
        return cols

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        vector1_values = frame[self.vector1.get_name()] if isinstance(self.vector1, Looper) else self.vector1
        vector2_values = frame[self.vector2.get_name()] if isinstance(self.vector2, Looper) else self.vector2
        frame['__start__'] = np.abs(vector1_values-vector2_values)
        frame['__end__'] = vector1_values+vector2_values
        frame[name] = frame.apply(lambda row: list(np.arange(row['__start__'], row['__end__'] + 1)), axis=1)
        frame = frame.drop(columns=['__start__', '__end__'])
        if explode:
            frame = frame.explode(name)
            frame = frame.astype({name: float})
        return frame

class Intersection(Looper):
    def __init__(self, *args:Looper):
        super().__init__()
        self.loopers = args
        for i, looper in enumerate(self.loopers):
            assert not isinstance(looper, Dummy)
            if not looper.is_name_set():
                looper.set_name(get_unique_name())


    def get_directly_dependent_columns(self)->set:
        return set().union(*[looper.get_directly_dependent_columns() for looper in self.loopers])

    def get_dependent_columns(self)->set:
        return set().union(*[looper.get_dependent_columns() for looper in self.loopers])

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()

        for looper in self.loopers:
            frame = looper.fill_frame(frame, explode=False)
        cols = [looper.get_name() for looper in self.loopers]
        frame[name] = frame.apply(lambda row: list(reduce(np.intersect1d, [row[col] for col in cols])), axis=1)

        if explode:
            frame = frame.explode(name)
            if frame[name].isna().any():
                logging.warning(f"NaN values found when computing intersection for {name}.")

            frame = frame.drop(columns=cols).dropna(subset=[name])
            frame = frame.astype({name: float})

        return frame


class Sum(Looper):
    def __init__(self, left:Looper, right:Looper):
        super().__init__()
        self.left = left
        self.right = right
        if not self.left.is_name_set():
            self.left.set_name(get_unique_name())
        if not self.right.is_name_set():
            self.right.set_name(get_unique_name())

    def get_directly_dependent_columns(self)->set:
        return self.left.get_directly_dependent_columns().union(self.right.get_directly_dependent_columns())

    def get_dependent_columns(self)->set:
        return self.left.get_dependent_columns().union(self.right.get_dependent_columns())

    def fill_frame(self, frame: pd.DataFrame, explode=True, default_name:str=None) -> pd.DataFrame:
        name = self.get_name() if default_name is None else default_name
        frame = self.left.fill_frame(frame, explode=False)
        frame = self.right.fill_frame(frame, explode=False)
        frame[name] = frame[self.left.get_name()] + frame[self.right.get_name()]

        if explode:
            frame = frame.explode(name)
            if frame[name].isna().any():
                logging.warning(f"NaN values found when computing intersection for {name}.")

            frame = frame.drop(columns=[self.left.get_name(),
                                        self.right.get_name()
                                        ]).dropna(subset=[name])
            frame = frame.astype({name: float})

        return frame

class Difference(Looper):
    def __init__(self, left:Looper, right:Looper):
        super().__init__()
        self.left = left
        self.right = right
        if not self.left.is_name_set():
            self.left.set_name(get_unique_name())
        if not self.right.is_name_set():
            self.right.set_name(get_unique_name())

    def get_directly_dependent_columns(self)->set:
        return self.left.get_directly_dependent_columns().union(self.right.get_directly_dependent_columns())

    def get_dependent_columns(self)->set:
        return self.left.get_dependent_columns().union(self.right.get_dependent_columns())

    def fill_frame(self, frame: pd.DataFrame, explode=True, default_name:str=None) -> pd.DataFrame:
        name = self.get_name() if default_name is None else default_name
        frame = self.left.fill_frame(frame, explode=False)
        frame = self.right.fill_frame(frame, explode=False)
        frame[name] = frame[self.left.get_name()] - frame[self.right.get_name()]

        if explode:
            frame = frame.explode(name)
            if frame[name].isna().any():
                logging.warning(f"NaN values found when computing intersection for {name}.")

            frame = frame.drop(columns=[self.left.get_name(),
                                        self.right.get_name()
                                        ]).dropna(subset=[name])
            frame = frame.astype({name: float})

        return frame
