import types
import inspect
import functools

def with_extra_args_support(optimizer):
    original_step = optimizer.step
    valid_params = inspect.signature(original_step).parameters

    @functools.wraps(original_step)
    def new_step(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return original_step(*args, **filtered_kwargs)

    optimizer.step = types.MethodType(new_step, optimizer)
    return optimizer
