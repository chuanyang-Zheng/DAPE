# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
import torch.nn as nn


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, dim, max_seq_len, base=10000, precision=torch.half, save_inv_freqs=False
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        self.max_seq_len = max_seq_len
        self.base = base
        self.dim = dim

        # precompute cos_cached, sin_cached in fp32
        cos_cached, sin_cached, inv_freq = self._prepare_cache(
            max_seq_len, precision, base
        )

        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def _prepare_cache(self, seq_len, precision, base):
        # precompute cos_cached, sin_cached in fp32
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_cached = emb.cos()[:, None, None, :]
        sin_cached = emb.sin()[:, None, None, :]

        return (
            cos_cached.to(precision),
            sin_cached.to(precision),
            inv_freq.to(precision),
        )

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        assert seq_len <= self.max_seq_len

        if seq_len != self.max_seq_len:
            # y, z, _ = self._prepare_cache(seq_len, self.precision, self.base)
            return (
                self.cos_cached[:seq_len, ...].to(x.device),
                self.sin_cached[:seq_len, ...].to(x.device),
            )
        else:
            return self.cos_cached.to(x.device), self.sin_cached.to(x.device)


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, offset: int = 0
):  # jitting fails with bf16
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)






class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1, noise_max_length=32768):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
                 mp_rank * self.slice_size: (mp_rank + 1) * self.slice_size
                 ]
        self.register_buffer("slopes", slopes)
        self.noise_max_length = noise_max_length
        print(slopes.dtype)

        self.cache_matrix_my_len = 2000
        self.cached_matrix_my = -torch.tril(
            torch.arange(self.cache_matrix_my_len)
            .view(self.cache_matrix_my_len, 1)
            .repeat(1, self.cache_matrix_my_len)
            + torch.arange(0, -self.cache_matrix_my_len, -1)
        )

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + self._get_slopes(2 * closest_power_of_2)[0::2][
                      : n - closest_power_of_2
                      ]
            )

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            # print(a)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a


class AliBi_C(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1, noise_max_length=32768, mlp_width=32):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
                 mp_rank * self.slice_size: (mp_rank + 1) * self.slice_size
                 ]
        self.register_buffer("slopes", slopes)
        self.noise_max_length = noise_max_length
        self.mlp2 = nn.Sequential(
            nn.Linear(num_heads * 2, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))

        self.cache_matrix_my_len = 2000
        self.cached_matrix_my = -torch.tril(
            torch.arange(self.cache_matrix_my_len)
            .view(self.cache_matrix_my_len, 1)
            .repeat(1, self.cache_matrix_my_len)
            + torch.arange(0, -self.cache_matrix_my_len, -1)
        )

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + self._get_slopes(2 * closest_power_of_2)[0::2][
                      : n - closest_power_of_2
                      ]
            )


    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            # print(a)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        x_a_bias = torch.cat((x, torch.tile(a, (x.shape[0], 1, 1, 1))), dim=1)
        # print(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 2, 3, 1))
        x_a_bias = self.mlp2(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 3, 1, 2))

        return x + a + x_a_bias


class Alibi_DAPEV2(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1, noise_max_length=32768, mlp_width=32,neox_args=None):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
                 mp_rank * self.slice_size: (mp_rank + 1) * self.slice_size
                 ]
        self.register_buffer("slopes", slopes)
        self.noise_max_length = noise_max_length
        # self.mlp2 = nn.Sequential(
        #     nn.Conv2d(num_heads * 2, mlp_width, (1, 3), (1, 1), (0, 1), (1, 1)),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(mlp_width, num_heads, (1, 3), (1, 1), (0, 1), (1, 1)))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(num_heads * 2, neox_args.mlp_width, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(neox_args.mlp_width, num_heads, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)))

        self.cache_matrix_my_len = 2000
        self.cached_matrix_my = -torch.tril(
            torch.arange(self.cache_matrix_my_len)
            .view(self.cache_matrix_my_len, 1)
            .repeat(1, self.cache_matrix_my_len)
            + torch.arange(0, -self.cache_matrix_my_len, -1)
        )

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + self._get_slopes(2 * closest_power_of_2)[0::2][
                      : n - closest_power_of_2
                      ]
            )


    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            # print(a)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        x_a_bias = torch.cat((x, torch.tile(a, (x.shape[0], 1, 1, 1))), dim=1)
        # print(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,2,3,1))
        x_a_bias = self.mlp2(torch.tril(x_a_bias))

        # x_a_bias=self.mlp2(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,3,1,2))

        return x + a + x_a_bias




class FIRE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6, noise_max_length=32768):
        super(FIRE, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, num_heads))
        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))
        self.noise_max_length = noise_max_length

        # Initialize L (threshold)

        self.init_L = nn.Parameter(torch.tensor(init_L // 2), requires_grad=False)
        # Learn a multiplier to L

        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

        self.cached_matrix = None
        self.cached_seq_len = None

    def forward(self, x: torch.Tensor):
        seq_length = x.size(2)

        # positions = torch.arange(seq_length, dtype=torch.float, device=x.device)

        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix
        # print(rel_distance)

        # rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(rel_distance), dim=-1)[0]
        # print(rel_distance_max)

        pos_normalizer = torch.max(rel_distance_max, threshold)

        pos_normalizer = pos_normalizer[:, None]

        # self.c=self.c.to(rel_distance.device)
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1
                                 )

        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)
        # print(normalized_distance)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))

        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        # print(x)
        # print(fire_bias)

        return x + fire_bias


class FIRE_C(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6, noise_max_length=32768):
        super(FIRE_C, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))
        # Initialize c (log transformation parameter)

        self.mlp2 = nn.Sequential(
            nn.Linear(num_heads * 2, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))
        self.c = nn.Parameter(torch.tensor(init_c))
        self.noise_max_length = noise_max_length

        # Initialize L (threshold)

        self.init_L = nn.Parameter(torch.tensor(init_L // 2), requires_grad=False)
        # Learn a multiplier to L

        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

        self.cached_matrix = None
        self.cached_seq_len = None

    def forward(self, x: torch.Tensor):
        seq_length = x.size(2)

        # positions = torch.arange(seq_length, dtype=torch.float, device=x.device)

        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix

        rel_distance = rel_distance.to(x.device)
        # print(rel_distance)

        # rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(rel_distance), dim=-1)[0]
        # print(rel_distance_max)

        pos_normalizer = torch.max(rel_distance_max, threshold)
        # print(threshold)

        pos_normalizer = pos_normalizer[:, None]

        # self.c=self.c.to(rel_distance.device)
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1
                                 )

        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)
        # print(normalized_distance)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))

        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        x_fire_bias = torch.cat((x, torch.tile(fire_bias, (x.shape[0], 1, 1, 1))), dim=1)
        x_fire_bias = x_fire_bias.permute(0, 2, 3, 1)
        x_fire_bias = self.mlp2(x_fire_bias)
        x_fire_bias = x_fire_bias.permute(0, 3, 1, 2)

        return x + fire_bias + x_fire_bias


class FIRE_DAPEV2(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6, noise_max_length=32768, neox_args=None):
        super(FIRE_DAPEV2, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads))
        # Initialize c (log transformation parameter)

        # self.mlp2 = nn.Sequential(
        #     nn.Conv2d(num_heads * 2, mlp_width, (1, 3), (1, 1), (0, 1), (1, 1)),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(mlp_width, num_heads, (1, 3), (1, 1), (0, 1), (1, 1)))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(num_heads * 2, neox_args.mlp_width, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(neox_args.mlp_width, num_heads, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)))
        self.c = nn.Parameter(torch.tensor(init_c))
        self.noise_max_length = noise_max_length

        # Initialize L (threshold)

        self.init_L = nn.Parameter(torch.tensor(init_L // 2), requires_grad=False)
        # Learn a multiplier to L

        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

        self.cached_matrix = None
        self.cached_seq_len = None

    def forward(self, x: torch.Tensor):
        seq_length = x.size(2)

        # positions = torch.arange(seq_length, dtype=torch.float, device=x.device)

        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix

        rel_distance = rel_distance.to(x.device)
        # print(rel_distance)

        # rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(rel_distance), dim=-1)[0]
        # print(rel_distance_max)

        pos_normalizer = torch.max(rel_distance_max, threshold)
        # print(threshold)

        pos_normalizer = pos_normalizer[:, None]

        # self.c=self.c.to(rel_distance.device)
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1
                                 )

        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)
        # print(normalized_distance)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))

        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        # print(torch.tril(fire_bias))
        # print(fire_bias)
        x_fire_bias = torch.cat((x, torch.tile(fire_bias, (x.shape[0], 1, 1, 1))), dim=1)
        # x_fire_bias=x_fire_bias.permute(0,2,3,1)

        x_fire_bias = self.mlp2(torch.tril(x_fire_bias))

        # x_fire_bias=self.mlp2(x_fire_bias)
        # x_fire_bias = x_fire_bias.permute(0, 3, 1, 2)

        return x + fire_bias + x_fire_bias


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]
class RelativePositionalEncodingLayer(torch.nn.Module):

    def __init__(self, num_heads, num_hiddens, noise_max_length=32768, max_time=10_000, mlp_width=32, scale_factor=None,
                 train_length=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.noise_max_length = noise_max_length
        self.max_time = max_time
        self.scale_factor = scale_factor
        self.r_w_bias = nn.Parameter(torch.randn(self.num_heads, self.num_hiddens) * 0.02)
        self.r_r_bias = nn.Parameter(torch.randn(self.num_heads, self.num_hiddens) * 0.02)
        self.r_net = nn.Linear(num_hiddens * num_heads, num_hiddens * num_heads, bias=False)
        self.pos_emb = PositionalEmbedding(num_hiddens * num_heads)
        self.train_length = train_length // 2

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, queries, keys):
        k_seq_len, batch_size, _, _ = keys.shape
        # print(keys.shape)
        pos_seq = torch.arange(k_seq_len - 1, -1, -1.0, device=queries.device,
                               dtype=queries.dtype)
        # pos_seq.clamp_(max=self.train_length)
        pos_emb = self.pos_emb(pos_seq)

        hiddens = self.num_hiddens * self.num_heads
        # print("Query"+str(torch.max(queries)))
        # print("Key"+str(torch.max(keys)))

        # Content logits.
        # content_logits = torch.einsum('bthd,bThd->bhtT', queries + self.content_bias, keys)
        rw_head_q = queries + self.r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, keys))
        rr_head_q = queries + self.r_r_bias
        r_head_k = self.r_net(pos_emb)
        r_head_k = r_head_k.view(k_seq_len, self.num_heads, self.num_hiddens)
        # print(r_head_k.shape)
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)
        # print(content_logits.shape)

        # print("AC"+str(torch.max(AC)))
        # print("BD"+str(torch.max(AC)))
        assert AC.shape == BD.shape
        attention = (AC + BD) * self.scale_factor
        attention = attention.permute(2, 3, 0, 1)
        # print(torch.max(attention))

        return attention




class ParallelKerpleLog(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length

        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        # assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return nn.Parameter(torch.ones(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return nn.Parameter(torch.rand(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)

        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None

        self.cache_matrix_my_len = 2000
        self.cache_matrix_my = torch.tril(
            torch.arange(self.cache_matrix_my_len)
            .view(self.cache_matrix_my_len, 1)
            .repeat(1, self.cache_matrix_my_len)
            + torch.arange(0, -self.cache_matrix_my_len, -1)
        )

    def stats(self):
        def get_stats(name, obj):
            return {name + '_mean': obj.mean().detach().cpu(),
                    name + '_std': obj.std().detach().cpu(),
                    name + '_max': obj.max().detach().cpu(),
                    name + '_min': obj.min().detach().cpu()}

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"

            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        return x + bias


class ParallelKerpleLog_C(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
            layer_number
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length
        self.layer_number_visualization = layer_number

        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        # assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return nn.Parameter(torch.ones(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return nn.Parameter(torch.rand(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)

        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_heads_per_partition * 2, neox_args.mlp_width),
            nn.LeakyReLU(),
            nn.Linear(neox_args.mlp_width, self.num_heads_per_partition))

        self.cache_matrix_my_len = 2000
        self.cache_matrix_my = torch.tril(
            torch.arange(self.cache_matrix_my_len)
            .view(self.cache_matrix_my_len, 1)
            .repeat(1, self.cache_matrix_my_len)
            + torch.arange(0, -self.cache_matrix_my_len, -1)
        )

    def stats(self):
        def get_stats(name, obj):
            return {name + '_mean': obj.mean().detach().cpu(),
                    name + '_std': obj.std().detach().cpu(),
                    name + '_max': obj.max().detach().cpu(),
                    name + '_min': obj.min().detach().cpu()}

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"

            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        x_a_bias = torch.cat((x, torch.tile(bias, (x.shape[0], 1, 1, 1))), dim=1)
        # print(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 2, 3, 1))
        x_a_bias = self.mlp2(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 3, 1, 2))


        return x + bias + x_a_bias



class Kerple_DAPEV2(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
            layer_number
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length
        self.layer_number_visualization = layer_number

        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        # assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return nn.Parameter(torch.ones(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return nn.Parameter(torch.rand(
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )[:, None, None] * scale)

        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None
        self.mlp2 = nn.Sequential(
            nn.Conv2d(self.num_heads_per_partition * 2, neox_args.mlp_width, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(neox_args.mlp_width, self.num_heads_per_partition, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)))

        self.cache_matrix_my_len = 2000
        self.cache_matrix_my = torch.tril(
            torch.arange(self.cache_matrix_my_len)
            .view(self.cache_matrix_my_len, 1)
            .repeat(1, self.cache_matrix_my_len)
            + torch.arange(0, -self.cache_matrix_my_len, -1)
        )

    def stats(self):
        def get_stats(name, obj):
            return {name + '_mean': obj.mean().detach().cpu(),
                    name + '_std': obj.std().detach().cpu(),
                    name + '_max': obj.max().detach().cpu(),
                    name + '_min': obj.min().detach().cpu()}

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                    seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"

            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        x_a_bias = torch.cat((x, torch.tile(bias, (x.shape[0], 1, 1, 1))), dim=1)
        # print(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,2,3,1))
        x_a_bias = self.mlp2(torch.tril(x_a_bias))

        return x + bias + x_a_bias




class Nope_C(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length

        self.cached_matrix = None
        self.cached_seq_len = None
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_heads_per_partition, neox_args.mlp_width),
            nn.LeakyReLU(),
            nn.Linear(neox_args.mlp_width, self.num_heads_per_partition))

    def forward(self, x):
        x_a_bias = x
        # print(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 2, 3, 1))
        x_a_bias = self.mlp2(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 3, 1, 2))

        return x + x_a_bias


class Nope_C_FFN(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length

        self.cached_matrix = None
        self.cached_seq_len = None
        # self.mlp2 = nn.Sequential(
        #     nn.Conv2d(self.num_heads_per_partition, neox_args.mlp_width, (1, 3), (1, 1), (0, 1), (1, 1)),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(neox_args.mlp_width, self.num_heads_per_partition, (1, 3), (1, 1), (0, 1), (1, 1)))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(self.num_heads_per_partition * 2, neox_args.mlp_width, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(neox_args.mlp_width, self.num_heads_per_partition, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)))

    def forward(self, x):
        x_a_bias = x
        # print(x_a_bias)
        x_a_bias = torch.tril(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,2,3,1))
        x_a_bias = self.mlp2(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,3,1,2))

        return x + x_a_bias



class DAPEV1(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length

        self.cached_matrix = None
        self.cached_seq_len = None
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_heads_per_partition, neox_args.mlp_width),
            nn.LeakyReLU(),
            nn.Linear(neox_args.mlp_width, self.num_heads_per_partition))

    def forward(self, x):
        x_a_bias = x
        # print(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 2, 3, 1))
        x_a_bias = self.mlp2(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 3, 1, 2))

        return x + x_a_bias


class DAPEV2(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
            self,
            neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        # self.model_parallel_size = get_model_parallel_world_size()
        # self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        self.noise_max_length = neox_args.noise_seq_length

        self.cached_matrix = None
        self.cached_seq_len = None
        # self.mlp2 = nn.Sequential(
        #     nn.Conv2d(self.num_heads_per_partition, neox_args.mlp_width, (1, 3), (1, 1), (0, 1), (1, 1)),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(neox_args.mlp_width, self.num_heads_per_partition, (1, 3), (1, 1), (0, 1), (1, 1)))

        self.mlp2 = nn.Sequential(
            nn.Conv2d(self.num_heads_per_partition * 2, neox_args.mlp_width, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(neox_args.mlp_width, self.num_heads_per_partition, (1, neox_args.dapev2_kernel), (1, 1),
                      (0, neox_args.dapev2_kernel // 2), (1, 1)))



    def forward(self, x):
        x_a_bias = x
        # print(x_a_bias)
        x_a_bias = torch.tril(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,2,3,1))
        x_a_bias = self.mlp2(x_a_bias)
        # x_a_bias=torch.permute(x_a_bias,(0,3,1,2))

        return x + x_a_bias


class CoPE(nn.Module):
    def __init__(self, npos_max=64, head_dim=12):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(
            torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        gates = torch.sigmoid(attn_logits)
        gates = torch.tril(gates)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        cope_result = logits_ceil * w + logits_floor * (1 - w)

        attn_logits = attn_logits + cope_result

        return attn_logits
