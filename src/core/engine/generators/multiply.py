import numpy as np


def multiply(*args, is_complex=False):
    """
    DO NOT DELAY EVALUATION!
    DO NOT PUT multiply INSIDE DELAYED-EVALUATION FUNCTIONS/CALLABLES!

    ---

    multiplies all args in order. If arg is a callable, it can be short-circuited.

    usage:
    result = multiply(
        delta(i, 1),
        lambda: expensive_calculation(i),  # <- this will be evaluated only if previous terms are not 0
    )

    It is most efficient to put the most likely to be 0 terms first, like delta or 3j symbols.

    ==== NOTICE ON CALLABLES ====

    multiply should be safe to use in loops on its own.
    It returns immediately, during the loop iteration, so the fact that lambdas in the arguments
    use references to the variables and not the actual values will not cause errors.

    The only scenario where it can cause an error is if the multiply function itself is evaluated
    with delay.

    If multiply is to be used in a callable, one should proceed with utmost caution, as it can cause
    hard-to-detect errors, as illustrated below.

    multiply has to be evaluated IMMEDIATELY, or the innermost lambda should encapsulate
    the entire scope where the local variables change.

    The following code illustrates the problem:

    callables = []
    for i in [0, 1]:
        callables.append(lambda : multiply(
            lambda: delta(i, 1)  # This will be evaluated for i==1 both times!
        ))
    results = [c() for c in callables]

    Same goes for the following:

    callables = []
    for i in [0, 1]:
        callables.append(lambda : multiply(
            delta(i, 1) # This will be evaluated for i==1 both times!
        ))
    results = [c() for c in callables]
    """

    result = np.array([1.0], dtype=np.float64) if not is_complex else np.array([1.0], dtype=np.complex128)
    for arg in args:
        value = arg()

        if not isinstance(value, np.ndarray):  # Short circuit only scalars for now
            if value == 0:
                return 0
        result = result * value

    return result
