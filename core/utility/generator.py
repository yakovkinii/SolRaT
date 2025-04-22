from core.utility.math import δ
from core.utility.python import range_inclusive, triangular, projection


def nested_loops(variables: dict, arguments: dict = None):
    """
    :param variables: dict {'parameter_name':'parameter iterable'}
    :param arguments: dict {'argument_name':'argument'}
    :return: generator

    usage:

    for k, q in nested_loops(
        variables={
            "k": "range(p0)",
            "q": "range_inclusive(-k, k)",
        },
        arguments={"p0": p0},
    ):
        R += k * abs(q)

    alternatively:

    for k, q in nested_loops(
        variables={
            "k": f"range({p0})",
            "q": "range_inclusive(-k, k)",
        },
    ):
        R += k * abs(q)

    """
    if arguments is None:
        arguments = {}

    code = "def _loop(" + ", ".join(arguments.keys()) + "):\n"
    tabs = 1
    for variable, variable_range in variables.items():
        code += "\t" * tabs + f"for {variable} in {variable_range}:\n"
        tabs += 1
    code += "\t" * tabs + "yield " + ", ".join(variables.keys())

    input_scope = {
        "range_inclusive": range_inclusive,
        "triangular": triangular,
        "projection": projection,
    }
    output_scope = {}

    exec(code, input_scope, output_scope)
    return output_scope["_loop"](*arguments.values())


def Σ(expression: callable, **kwargs):
    """
    :param expression: expression to sum
    :param kwargs: indexes to loop through, with iterables for their values.
    values can be callable
    :return:

    usage:
    R2 = Σ(
        lambda K, Q, Kʹ, Qʹ: K * abs(Q) + Kʹ * abs(Qʹ),
        K=f"range({p0})",
        Kʹ=f"range({p0})",
        Q="range_inclusive(-K, K)",
        Qʹ="range_inclusive(-Kʹ, Kʹ)",
    )

    """
    tabs = 0
    code = "result = 0\n"
    for variable, variable_range in kwargs.items():
        code += "\t" * tabs + f"for {variable} in {variable_range}:\n"
        tabs += 1
    code += "\t" * tabs + "result += expression(" + ", ".join([f"{key}={key}" for key in kwargs.keys()]) + ")"

    input_scope = {
        "expression": expression,
        "range_inclusive": range_inclusive,
        "triangular": triangular,
        "projection": projection,
    }
    output_scope = {}

    exec(code, input_scope, output_scope)
    return output_scope["result"]


def multiply(*args):
    """
    DO NOT PUT multiply INSIDE LAMBDAS OR OTHER DELAYED-EVALUATION FUNCTIONS/CALLABLES!

    ---

    multiplies all args in order. If arg is a callable, it can be short circuited.
    :param args:
    :return:

    usage:
    result = multiply(
        δ(i, 1),
        lambda: expensive_calculation(i),  # <- this will be evaluated only if previous terms are not 0
    )

    It is most efficient to put the most likely to be 0 terms first, like δ or 3j symbols.

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
            lambda: δ(i, 1)  # This will be evaluated for i==1 both times!
        ))
    results = [c() for c in callables]

    Same goes for the following:

    callables = []
    for i in [0, 1]:
        callables.append(lambda : multiply(
            δ(i, 1) # This will be evaluated for i==1 both times!
        ))
    results = [c() for c in callables]

    The only safe way is when the entire scope of the changing variables is within the innermost lambda.
    Following code is not tested, but should work correctly:

    results = []
    non_changing_variable = 1
    command = lambda: '''
    for i in [0, 1]:
        results.append(multiply(
            lambda: δ(non_changing_variable, 1),
            lambda: δ(i, 1)  # This should be OK, as multiply will be evaluated between the loop iterations
        ))
    '''
    exec(command())


    """
    result = 1
    for arg in args:
        if callable(arg):
            value = arg()
        else:
            value = arg

        if value == 0:
            return 0
        result *= value
    return result


def n_proj(*args):
    # 2 n + 1. If multiple arguments - multiply
    result = 1
    for arg in args:
        result *= 2 * arg + 1
    return result


if __name__ == "__main__":
    p0 = 40
    R = 0
    for k, q in nested_loops(
        variables={
            "k": "range(p0)",
            "q": "range_inclusive(-k, k)",
        },
        arguments={"p0": p0},
    ):
        R += k * abs(q)
    # print(R)

    # ======

    R1 = 0
    for k, q in nested_loops(
        {
            "k": f"range({p0})",
            "q": "range_inclusive(-k, k)",
        }
    ):
        R1 += k * abs(q)
    # print(R1)

    # ======
    import time

    t0 = time.perf_counter()
    R = 0
    for k in range(p0):
        for q in range_inclusive(-k, k):
            for k_prime in range(p0):
                for q_prime in range_inclusive(-k_prime, k_prime):
                    R += k * abs(q) + k_prime * abs(q_prime)
    t1 = time.perf_counter()
    print(R)
    print(t1 - t0)

    # ======
    t0 = time.perf_counter()

    R2 = Σ(
        lambda K, Q, Kʹ, Qʹ: K * abs(Q) + Kʹ * abs(Qʹ),
        K=f"range({p0})",
        Kʹ=f"range({p0})",
        Q="range_inclusive(-K, K)",
        Qʹ="range_inclusive(-Kʹ, Kʹ)",
    )

    t1 = time.perf_counter()
    print(R2)
    print(t1 - t0)

    # ======
    def expensive_calculation(x):
        print(f"expensive_calculation evaluated for {x=}")
        return x

    for i in [0, 1]:
        R3 = multiply(
            δ(i, 1),
            lambda: expensive_calculation(i),
        )
        print(R3)

    # Test latex
    Kʹ = 0
    Jᵤ = 1
    Jₗ = 1
    𝔗 = 1
    𝚃 = 1
    ρ = 1
    ηᴬ = 1
    ηA = 1
    ᆝ = 1
    d3 = 3
    ηₐ = 1
    ϵᴱ = 1
    εˢ = 1
    ρˢ = 1
    ν = 1
    𝛎 = 1
    𝝂 = 1
    𝝼 = 1
    𝞶 = 1
