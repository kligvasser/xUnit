import torch
import math

def get_exp_scheduler_with_warmup(optimizer, rampup_steps=5, sustain_steps=5):
    def lr_lambda(step):
        if step < rampup_steps:
            return min(1., 1.8 ** ((step - rampup_steps)))
        elif step < rampup_steps + sustain_steps:
            return 1.
        else:
            return max(0.1, 0.85 ** (step - rampup_steps - sustain_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def get_cosine_scheduler_with_warmup(optimizer, rampup_steps=3, sustain_steps=2, priod=5):
    def lr_lambda(step):
        if step < rampup_steps:
            return min(1., 1.5 ** ((step - rampup_steps)))
        elif step < rampup_steps + sustain_steps:
            return 1.
        else:
            return max(0.1, (1 + math.cos(math.pi * (step - rampup_steps - sustain_steps) / priod)) / 2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-4)
    lr_scheduler = get_exp_scheduler_with_warmup(optimizer)

    lrs = []
    for i in range(25):
        lr_scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    plt.plot(lrs)
    plt.show()
    print(lrs)
