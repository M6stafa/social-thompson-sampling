

def exp_decay(init_value: float, decay_rate: float, decay_steps: int):
    def decay(step: int) -> float:
        return init_value * decay_rate ** (step / decay_steps)
    return decay
