import numpy as np


def multiply(*args, is_complex=False, is_scalar=False):
    r"""
    multiplies all args in order. If arg is a callable, it can be short-circuited.

    usage:

    .. code-block:: python

        result = multiply(
            delta(i, 1),
            lambda: expensive_calculation(i),  # <- this will be evaluated only if previous terms are not 0
        )

    It is most efficient to put the most likely to be 0 terms first, like delta or 3j symbols.

    ==== NOTE ON CALLABLES ====

    multiply is safe to use in loops on its own.
    It returns immediately, during the loop iteration, so the fact that lambdas in the arguments
    use references to the variables and not the actual values will not cause errors.

    The only scenario where it can cause an error is if the multiply function itself is evaluated
    with delay. This behavior is not specific to multiply, it follows from the behavior of delayed
    evaluation as implemented in python.

    The following code illustrates the potential problem:

    .. code-block:: python

        callables = []
        for i in [0, 1]:
            callables.append(lambda : multiply(
                lambda: delta(i, 1)  # This will be evaluated for i==1 both times!
            ))  # multiply should be evaluated here instead of being delayed.

        results = [c() for c in callables]

    """
    if is_scalar:
        result = 1.0
    else:
        result = np.array([1.0], dtype=np.float64) if not is_complex else np.array([1.0], dtype=np.complex128)

    for arg in args:
        value = arg()

        if not isinstance(value, np.ndarray):  # Short circuit only scalars for now
            if value == 0:
                return 0
        result = result * value

    return result
