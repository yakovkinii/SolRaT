from core.utility.python import range_inclusive, triangular, projection


def nested_loops(variables: dict, arguments: dict = None):
    """
    :param variables: dict {'parameter_name':'parameter iterable'}
    :param arguments: dict {'argument_name':'argument'}
    :return: generator
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
    tabs = 0
    code = "result = 0\n"
    for variable, variable_range in kwargs.items():
        code += "\t" * tabs + f"for {variable} in {variable_range}:\n"
        tabs += 1
    code += (
        "\t" * tabs
        + "result += expression("
        + ", ".join([f"{key}={key}" for key in kwargs.keys()])
        + ")"
    )

    input_scope = {
        "expression": expression,
        "range_inclusive": range_inclusive,
        "triangular": triangular,
        "projection": projection,
    }
    output_scope = {}

    exec(code, input_scope, output_scope)
    return output_scope["result"]


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
    print(t1-t0)

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
    print(t1-t0)

    # Test latex
    Kʹ = 0
    Jᵤ = 1
    Jₗ = 1
    𝔗 = 1
    𝚃 = 1
    ρ = 1
    ηᴬ = 1
    ηA = 1
    ηₐ = 1
    ϵᴱ = 1
    εˢ = 1
    ρˢ = 1
    ν = 1
    𝛎 = 1
    𝝂 = 1
    𝝼 = 1
    𝞶 = 1