"""
Wrappers to execute functions in parallel.

The wrappers all work on input / output file basis, where each call handlas the list
of files consecutively:
- 1st from input -> 1st output
- 2nd from input -> 2nd output
- etc.

The wrappers also handle single file calls where input and output are strings.

"""

import ray
from typing import Iterable, Dict, Callable


def parcall(
        iterable: Dict[str, Iterable],
        fun: Callable,
        num_cpus: int = 1,
        **fun_kwargs):
    """Execute function in parallel.

    The function ``fun`` must either take the input file path as first argument and, if ``out_files```
    is not ``None``, output file path as second argument. You can pass function arguments via
    ``fun_kwargs``.

    Parameters
    ----------
    iterable
        A dictionary of iterables (not a string) that are called in parallel with a key corresponding
        to the keyword argiument of ``fun``. The values must be iterable and of same length each.
    fun
        function to be executed in parallel, must take the input file path as first argument and if
        ``out_files`` is not ``None`` the output file path as second argument, You can pass function
        arguments via ``fun_kwargs``.
    fun_kwargs
        kwargs passed to ``fun``.
    num_cpus
        Number of cpus to use.

    """

    if not isinstance(iterable, dict):
        raise ValueError('Argument ``iterable`` must be of type ``dict``.')
    n_iter = -1
    first_key = ''
    for k, v in iterable.items():
        if isinstance(v, str) or not hasattr(v, '__iter__'):
            raise ValueError(
                f'In arg ``iterable``, value ``{v}`` of key ``{k}`` is either'
                'not an iterable or it is a string, which is both not allowed.')
        if n_iter == -1:
            n_iter = len(v)
            first_key = k
        else:
            if len(v) != n_iter:
                raise ValueError(
                    'All values in the dictionary ``iterable`` must be of the '
                    'same length, there is at least one mismatch, '
                    f'len(iterable[{first_key}]) = {n_iter} != len(iterable[{k}]) = {len(v)}.')

    iter_args = []
    for i in range(n_iter):
        single_arg = {}
        for k, v in iterable.items():
            single_arg.update({k: v[i]})
        iter_args.append(single_arg)

    @ray.remote
    def remote_fun(kwargs):
        return fun(**kwargs, **fun_kwargs)

    ray.init(num_cpus=num_cpus)

    try:
        results = ray.get(
            [remote_fun.remote(iter_arg)
                for iter_arg in iter_args]
        )

        if len(results) != len(iter_args):
            raise AssertionError('Something went wrong.')

        return results

    finally:
        ray.shutdown()
