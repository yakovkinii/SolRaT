import logging
from typing import Union
from functools import reduce
import numpy as np
import pandas as pd
import inspect

from abc import ABC, abstractmethod


class Looper(ABC):
    @abstractmethod
    def set_name(self, name:str):
        pass

    @abstractmethod
    def get_name(self)->str:
        pass

    @abstractmethod
    def get_directly_dependent_columns(self)->set:
        pass

    @abstractmethod
    def get_dependent_columns(self)->set:
        pass

    @abstractmethod
    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        pass


class Value(Looper):
    def __init__(self, value:Union[int, float]):
        self.value = value
        self.name = None

    def set_name(self, name: str):
        self.name = name

    def get_name(self)->str:
        assert self.name is not None
        return self.name

    def get_directly_dependent_columns(self)->set:
        return set()

    def get_dependent_columns(self)->set:
        return set()

    def fill_frame(self, frame: pd.DataFrame, explode=True) -> pd.DataFrame:
        name = self.get_name()
        frame[name] = self.value
        frame = frame.astype({name: float})
        return frame


class FromTo(Looper):
    def __init__(self, start:Union[Looper, int, float], end:Union[Looper, int, float]):
        self.start:Union[Looper, int, float] = start
        self.end:Union[Looper, int, float] = end
        self.name:Union[str, None] = None

    def set_name(self, name:str):
        self.name = name

    def get_name(self)->str:
        assert self.name is not None
        return self.name

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


# class Triangular(Looper):
#     def __init__(self, L:Union[Looper, int, float], S:Union[Looper, int, float]):
#

class Intersection(Looper):
    def __init__(self, *args:Looper):
        self.loopers = args
        for i, looper in enumerate(self.loopers):
            looper.set_name(f"__looper{i}__")
        self.name = None

    def set_name(self, name:str):
        self.name = name

    def get_name(self)->str:
        assert self.name is not None
        return self.name

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


