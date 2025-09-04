import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def compute_loss(x, y):
    return - (x ** 2 + 2 * y ** 2)


def optimize(param, optimizer, steps):
    param_list, loss_list = [], []
    for _ in range(steps):
        loss = compute_loss(param[0], param[1])
        loss.backward()

        param_list.append([param[0].item(), param[1].item()])
        loss_list.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()
    return param_list, loss_list


def visualize(params, loss, name):
    plt.cla()
    plt.figure(figsize=(4.8, 4.8))

    theta = np.linspace(0, 2*np.pi, 200)
    xs, ys = np.cos(theta), np.sin(theta)
    ys = ys / np.sqrt(2)

    plt.plot(xs, ys, c='lightgray')
    plt.plot(xs*2.0, ys*2.0, c='lightgray')

    xs = [p[0] for p in params]
    ys = [p[1] for p in params]
    plt.plot(xs, ys, marker='o', linestyle='-', label=name)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'sgd_max_{name}.png', dpi=300)


def main():
    steps = 20
    momentum_list = [0.0, 0.3, 0.6, 0.9]

    wd, save_name = (0.0, '')

    W, H = 4.5, 3.6

    all_loss = []
    for momentum in momentum_list:
        param = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))

        optimizer = torch.optim.SGD([param], lr=0.1, momentum=momentum, weight_decay=wd, maximize=True)

        params, loss = optimize(param, optimizer, steps)
        visualize(params, loss, name=f"momentum={momentum}{save_name}")
        all_loss.append(loss)

    plt.cla()
    plt.figure(figsize=(W, H))
    for i in range(len(momentum_list)):
        plt.plot(range(steps), all_loss[i], label=f'momentum={momentum_list[i]}{save_name}')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'sgd_max_loss{save_name}.png', dpi=300)


if __name__ == "__main__":
    main()
