import torch
import torch.optim as optim

def get_pda_multi_avg_fn(gamma=8.):
    """ Inspired by https://pytorch.org/docs/stable/optim.html#torch.optim.swa_utils.get_ema_multi_avg_fn """
    
    @torch.no_grad()
    def pda_update(pda_param_list, current_param_list, t):
        if torch.is_floating_point(pda_param_list[0]) or torch.is_complex(pda_param_list[0]):
            torch._foreach_lerp_(pda_param_list, current_param_list, (1 + gamma) / (t + gamma))
        else:
            # This is integer (or bool) so most probably a counter
            pass
            # for p_pda, p_model in zip(pda_param_list, current_param_list):
            #    p_pda.mul_(1 - (1 + gamma) / (t + gamma)).add_(p_model, alpha=(1 + gamma) / (t + gamma))
    
    return pda_update

def build_pda(gamma: float = 8.):
    return lambda model: optim.swa_utils.AveragedModel(model, multi_avg_fn=get_pda_multi_avg_fn(gamma=gamma), use_buffers=True)

def build_ema(decay: float = 0.999):
    return lambda model: optim.swa_utils.AveragedModel(model, avg_fn=optim.swa_utils.get_ema_multi_avg_fn(decay=decay), use_buffers=True)
