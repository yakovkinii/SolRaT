import inspect
import logging
from functools import reduce

import pandas as pd

from src.engine.generators.merge_loopers import Looper


def merge(df1, df2, on=None):
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    if len(on) == 0:
        return df1.merge(df2, how='cross')
    else:
        return df1.merge(df2, on=on, how='inner')


class Frame:
    def __init__(self, base_frame:pd.DataFrame=None, **kwargs: Looper):
        if base_frame is not None:
            self.base_frame = base_frame
        else:
            self.base_frame = pd.DataFrame(index=[0], columns=[])

        self.sub_frames = {}
        self.dependencies = {}
        self.dependencies_direct = {}
        self.sub_frame_orders = {}

        # df = self.base_frame.copy()
        for i, (looper_name, looper) in enumerate(kwargs.items()):
            looper.set_name(looper_name)
            dependent_cols = list(looper.get_directly_dependent_columns())
            sub_frame = self.construct_sub_frame(dependent_cols)
            sub_frame_filled = looper.fill_frame(sub_frame)
            self.sub_frames[looper_name] = sub_frame_filled
            self.dependencies[looper_name] = list(looper.get_dependent_columns())
            self.dependencies_direct[looper_name] = dependent_cols
            self.sub_frame_orders[looper_name] = i

            assert not sub_frame_filled[looper_name].isna().any()

        self.evaluation_frames = {}
        self.evaluation_dependencies = {}
        self.evaluation_dependencies_direct = {}
        self.evaluation_orders = {}

    def print_structure(self):
        print("Base frame:")
        print(self.base_frame)
        print("\nSub frames:")
        for name, frame in self.sub_frames.items():
            print(f"Sub frame '{name}':")
            print(frame)
            print(f"  Direct dependencies: {self.dependencies_direct[name]}")
            print(f"  All dependencies: {self.dependencies[name]}")

        print("\nEvaluation frames:")
        for name, frame in self.evaluation_frames.items():
            print(f"Evaluation frame '{name}':")
            print(frame)
            print(f"  Direct dependencies: {self.evaluation_dependencies_direct[name]}")
            print(f"  All dependencies: {self.evaluation_dependencies[name]}")

    def construct_full_frame(self):
        frames = [self.base_frame] + list(self.sub_frames.values())
        full_frame = reduce(lambda left, right: merge(left, right), frames)
        assert not full_frame.isna().any().any()
        assert len(full_frame) == len(full_frame.drop_duplicates())
        frames_evaluation = [full_frame]+list(self.evaluation_frames.values())
        result = reduce(lambda left, right: merge(left, right), frames_evaluation)
        return result.reset_index(drop=True)

    def get_frame_chain(self, name, names_to_trace):
        frame_names = []
        columns_to_trace = set(names_to_trace).intersection(set(self.dependencies[name]))
        frame_names.append(name)
        for col in self.dependencies_direct[name]:
            if all(tr not in self.dependencies[col] for tr in columns_to_trace) and col not in columns_to_trace:
                continue
            frame_names += self.get_frame_chain(col, names_to_trace)
        return frame_names

    def construct_sub_frame(self, columns, weights=False):
        if len(columns) == 0:
            return pd.DataFrame(index=[0], columns=[])

        frame_names = []
        for col in columns:
            frame_names.extend(self.get_frame_chain(col, columns))

        frame_names = list(set(frame_names))
        frame_names = sorted(frame_names, key=lambda x: self.sub_frame_orders[x])

        frames = []
        for name in frame_names:
            if weights or f"__weight__{name}" not in self.sub_frames[name].columns:
                frame = self.sub_frames[name]
            else:
                frame = self.sub_frames[name].drop(columns=[f"__weight__{name}"])
            frames.append(frame)

        result = reduce(lambda left, right: merge(left, right), frames)
        return result[columns].drop_duplicates().reset_index(drop=True)

    def evaluate(self, **kwargs)->pd.DataFrame:
        assert len(kwargs) == 1
        key, value = list(kwargs.items())[0]
        signature = inspect.signature(value)
        param_names = [p.name for p in signature.parameters.values()]
        sub_frame = self.construct_sub_frame(param_names)

        arguments = {name: sub_frame[name].values.reshape(-1,1) for name in param_names}
        sub_frame[key] = value(**arguments)
        self.evaluation_frames[key] = sub_frame
        self.evaluation_dependencies_direct[key] = param_names
        self.evaluation_dependencies[key] = list(set(param_names).union(*[set(self.dependencies[p]) for p in param_names]))
        self.evaluation_orders[key] = len(self.sub_frame_orders)
        return sub_frame

    # def drop(self, column):
    #     for name, sub_frame in self.sub_frames.items():
    #         if column in sub_frame.columns:
    #             sub_frame.drop(columns=[column], inplace=True)
    #     del self.sub_frames[column]
    #     del self.dependencies[column]
    #     del self.dependencies_direct[column]
    #     del self.sub_frame_orders[column]

    def reduce_frame(self, frame_name, column):
        frame = self.sub_frames[frame_name]
        weight_column = f"__weight__{frame_name}"
        frame_columns = [col for col in frame.columns if col not in [column, weight_column]]

        if weight_column not in frame.columns:
            frame[weight_column] = 1.0

        self.sub_frames[frame_name] = frame.groupby(frame_columns)[[weight_column]].agg({weight_column: 'sum'}).reset_index()
        self.dependencies[frame_name].remove(column)
        self.dependencies_direct[frame_name].remove(column)

    def reduce_evaluation_frames(self, column):
        evaluation_frames_to_reduce = [name for name, deps in self.evaluation_dependencies_direct.items() if column in deps]
        if len(evaluation_frames_to_reduce) == 0:
            return

        if len(evaluation_frames_to_reduce) == 1:
            self.reduce_single_evaluation_frame(evaluation_frames_to_reduce[0], column)
            return

        frames = [self.evaluation_frames[name] for name in evaluation_frames_to_reduce]
        frame = reduce(lambda left, right: merge(left, right), frames)
        product_column = "_*_".join(evaluation_frames_to_reduce)
        frame[product_column] = frame[evaluation_frames_to_reduce].prod(axis=1)

        for eval_name in evaluation_frames_to_reduce:
            frame = frame.drop(columns=[eval_name])

        self.evaluation_frames[product_column] = frame
        self.evaluation_dependencies_direct[product_column] = list(set().union(*[set(self.evaluation_dependencies_direct[name]) for name in evaluation_frames_to_reduce]))
        self.evaluation_dependencies[product_column] = list(set().union(*[set(self.evaluation_dependencies[name]) for name in evaluation_frames_to_reduce]))
        self.evaluation_orders[product_column] = max([self.evaluation_orders[name] for name in evaluation_frames_to_reduce])

        for eval_name in evaluation_frames_to_reduce:
            del self.evaluation_frames[eval_name]
            del self.evaluation_dependencies[eval_name]
            del self.evaluation_dependencies_direct[eval_name]
            del self.evaluation_orders[eval_name]

        self.reduce_single_evaluation_frame(product_column, column)

    def reduce_single_evaluation_frame(self, eval_name, column):
        frame = self.evaluation_frames[eval_name]

        frame_to_reduce = self.sub_frames[column]
        weight_column = f"__weight__{column}"
        if weight_column in frame_to_reduce.columns:
            print(f"Using weight column {weight_column} for reducing evaluation frame {eval_name} by {column}")
            print(frame_to_reduce)
            frame = frame.merge(frame_to_reduce, on=column)
            frame[eval_name] = frame[eval_name] * frame[weight_column]
            frame = frame.drop(columns=[weight_column])

        frame_columns = [col for col in frame.columns if col not in [column, eval_name]]
        if len(frame_columns) == 0:
            total = frame[eval_name].sum()
            self.evaluation_frames[eval_name] = pd.DataFrame({eval_name: [total]})
        else:
            self.evaluation_frames[eval_name] = frame.groupby(frame_columns)[[eval_name]].sum().reset_index()
        self.evaluation_dependencies[eval_name].remove(column)
        self.evaluation_dependencies_direct[eval_name].remove(column)

    def reduce(self, column):
        assert column in self.sub_frames, f"Column {column} not found in sub_frames"
        assert len(self.dependencies[column]) == 0, f"Cannot reduce column {column} with dependencies {self.dependencies[column]}"

        logging.info(f"Reducing {column}")

        for frame_name, frame in self.sub_frames.items():
            if frame_name == column:
                continue
            if column in self.dependencies_direct[frame_name]:
                self.reduce_frame(frame_name, column)

        self.reduce_evaluation_frames(column)

        del self.sub_frames[column]
        del self.dependencies[column]
        del self.dependencies_direct[column]
        del self.sub_frame_orders[column]
