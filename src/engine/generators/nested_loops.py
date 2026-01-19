"""
TODO
TODO  This file needs improved documentation.
TODO
"""

from src.engine.functions.looping import fromto, intersection, projection, triangular


def nested_loops(**kwargs):
    """
    :param kwargs: iterables to loop through, with iterables for their values.
    :return: generator

    usage:

    for k, q in nested_loops(k=f"range({p0})", q="fromto(-k, k)"):
        R += k * abs(q)

    The user should maintain the order of operands!!!
    """

    variables = kwargs

    code = "def _loop():\n"
    tabs = 1
    for variable, variable_range in variables.items():
        code += "\t" * tabs + f"for {variable} in {variable_range}:\n"
        tabs += 1
    code += "\t" * tabs + "yield " + ", ".join(variables.keys())

    input_scope = {
        "fromto": fromto,
        "triangular": triangular,
        "projection": projection,
        "intersection": intersection,
    }
    output_scope = {}

    exec(code, input_scope, output_scope)
    return output_scope["_loop"]()
