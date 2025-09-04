import time

from einops import rearrange
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class Attention(nn.Module):
    """https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MHA(nn.Module):

    def __init__(self, dim, head):
        super().__init__()
        self.mha = Attention(dim=dim, heads=head)

    def forward(self, x):
        return self.mha(x)


@torch.no_grad()
def profile(batch_size=1, seq_len=512, head=8, dim=768, avg=500):
    x = torch.randn(batch_size, seq_len, dim).cuda()
    mha = MHA(dim, head).cuda().eval()

    flops = FlopCountAnalysis(mha, x).total()

    time_list, memory_list = [], []
    for _ in range(avg):
        start = time.time()
        mha(x)
        time_list.append(time.time() - start)

        # we use max memory allocated to measure the peak memory usage; this is
        # fine for current profiling as memory usage is non-decreasing given
        # the same model and increasing input size
        memory_list.append(torch.cuda.max_memory_allocated() / 1024**3)

    wall_clock_time = np.mean(time_list)
    wall_clock_time_error = np.std(time_list) / (avg ** 0.5)

    memory = np.mean(memory_list)
    memory_error = np.std(memory_list) / (avg ** 0.5)

    return flops, memory, memory_error, wall_clock_time, wall_clock_time_error


def main():
    seq_len_list = [10, 100, 1_000, 10_000]

    flops_list, memory_list, memory_errors, wall_clock_time_list, wall_clock_time_errors = \
        [], [], [], [], []
    for seq_len in seq_len_list:
        flops, memory, memory_error, wall_clock_time, wall_clock_time_error = \
            profile(batch_size=1, seq_len=seq_len)
        flops_list.append(flops)
        memory_list.append(memory)
        memory_errors.append(memory_error)
        wall_clock_time_list.append(wall_clock_time)
        wall_clock_time_errors.append(wall_clock_time_error)

    H, W = 4.5, 3.6

    plt.figure(figsize=[H, W])
    plt.plot([0, 1, 2, 3], flops_list, '^-', label='FLOPs', c='#0082FB')
    plt.gca().set_xticks([0, 1, 2, 3], ['10', '100', '1K', '10K'])
    plt.gca().set_xlabel('Sequence Length')
    plt.gca().set_ylabel('FLOPs')
    plt.tight_layout()
    plt.savefig('flops.png', dpi=300)

    plt.cla()
    plt.figure(figsize=[H, W])
    plt.errorbar(
        [0, 1, 2, 3], memory_list, yerr=memory_errors,
        fmt='^-', label='Memory', c='#0082FB')
    plt.gca().set_xticks([0, 1, 2, 3], ['10', '100', '1K', '10K'])
    plt.gca().set_xlabel('Sequence Length')
    plt.gca().set_ylabel('Memory (GB)')
    plt.tight_layout()
    plt.savefig('memory.png', dpi=300)

    plt.cla()
    plt.figure(figsize=[H, W])
    plt.errorbar(
        [0, 1, 2, 3], wall_clock_time_list, yerr=wall_clock_time_errors,
        fmt='^-', label='Wall Clock Time', c='#0082FB')
    plt.gca().set_xticks([0, 1, 2, 3], ['10', '100', '1K', '10K'])
    plt.gca().set_xlabel('Sequence Length')
    plt.gca().set_ylabel('Seconds (Avg over 500 runs)')
    plt.tight_layout()
    plt.savefig('time.png', dpi=300)


if __name__ == "__main__":
    main()
