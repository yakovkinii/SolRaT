from tqdm import tqdm

from src.core.engine.functions.looping import fromto, triangular, projection, intersection


def summate(expression: callable, tqdm_level=1, **kwargs):
    """
    :param expression: expression to sum
    :param tqdm_level: Number of levels of tqdm to show.
    For tqdm_level>1 multiline editing support is needed from console.
    :param kwargs: indexes to loop through, with iterables for their values.
    values can be callable
    :return:

    usage:
    R2 = summate(
        lambda K, Q, Kʹ, Qʹ: K * abs(Q) + Kʹ * abs(Qʹ),
        K=f"range({p0})",
        Kʹ=f"range({p0})",
        Q="fromto(-K, K)",
        Qʹ="fromto(-Kʹ, Kʹ)",
    )

    Order of summation indexes in the first argument is not critical: the lambda is called using kwargs.

    """
    tabs = 0
    code = "result = 0\n"
    for variable, variable_range in kwargs.items():
        if tabs < tqdm_level:
            code += "\t" * tabs + f"for {variable} in tqdm({variable_range}, leave=False):\n"
        else:
            code += "\t" * tabs + f"for {variable} in {variable_range}:\n"
        tabs += 1
    code += "\t" * tabs + "result = result + expression(" + ", ".join([f"{key}={key}" for key in kwargs.keys()]) + ")"

    input_scope = {
        "expression": expression,
        "fromto": fromto,
        "triangular": triangular,
        "projection": projection,
        "intersection": intersection,
        "tqdm": tqdm,
    }
    output_scope = {}

    exec(code, input_scope, output_scope)
    return output_scope["result"]
