# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(threshold=9999999999)
def mm_norm(x1, x2, eps=1e-8):
    assert len(x2.size())==2
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = x2.transpose(0,1).norm(p=2, dim=-1, keepdim=True)

    mm=torch.matmul(x1, x2)
    norm=torch.matmul(w1, w2.transpose(0,1)).clamp(min=eps)

    return torch.div(mm, norm)

class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        hard=True,
        std=0,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first
        self.hard = hard

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        if std == 0:
            nn.init.uniform_(self.vars)
        else:
            nn.init.normal_(self.vars, mean=0, std=std)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)
        
        # for vector calibration
        # self.weight_proj_cos = torch.nn.Parameter(torch.eye(groups * num_vars), requires_grad=True)
        self.scaling_factor_for_vector = torch.nn.Parameter(torch.zeros(groups * num_vars), requires_grad=True)
        self.odim = groups * num_vars

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars**self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars**self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars**exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)

        # linear (original method)
        x = self.weight_proj(x)

        # cosine similarity (not work)
        # x = mm_norm(x,self.weight_proj_cos)

        # normalize
        avg = self.scaling_factor_for_vector.mean()

        # loss accelerator
        accel_factor = 10.0
        
        # vector scaling
        x = torch.matmul(x, (torch.eye(self.odim).to(x.device).to(x.dtype) + accel_factor * (self.scaling_factor_for_vector - avg).detach() * torch.eye(self.odim).to(x.device).to(x.dtype)))
        # x = torch.matmul(x, (torch.eye(self.odim).to(x.device).to(x.dtype) + (self.scaling_factor_for_vector - avg) * torch.eye(self.odim).to(x.device).to(x.dtype)))

        # train vector scaling function ('x2 size:', [1880, 640])
        x2 = torch.matmul(x.detach(), (torch.eye(self.odim).to(x.device).to(x.dtype) + accel_factor * (self.scaling_factor_for_vector - avg) * torch.eye(self.odim).to(x.device).to(x.dtype)))
        # x2 = torch.matmul(x, (torch.eye(self.odim).to(x.device).to(x.dtype) + (self.scaling_factor_for_vector - avg) * torch.eye(self.odim).to(x.device).to(x.dtype)))

        print_option=False
        if torch.randperm(10000)[0]==0:
            print('self.scaling_factor_for_vector:', self.scaling_factor_for_vector)
            print_option = True

        x2 = x2.mean(dim=0) # 640
        # print()
        # print('x2:', x2)
        x2 = torch.softmax(x2,dim=-1) * torch.log_softmax(x2,dim=-1)
        # print('x2 sm:', torch.softmax(x2,dim=-1))
        # print('x2 log sm:', torch.log_softmax(x2,dim=-1))
        loss_ent_max = x2.sum()
        result["loss_ent_max"] = loss_ent_max # TODO: loss ent를 codebook 별로 (두개로) 나누기
        # result["loss_ent_max"] = 0

        # print('x1', x.size()) # x1 torch.Size([1880, 640])
        x = x.view(bsz * tsz * self.groups, -1)
        # raise ValueError('x', x.size()) # 'x', torch.Size([3760, 320]

        with torch.no_grad():
            _, k = x.max(-1)
            hard_x = (
                x.new_zeros(*x.shape)
                .scatter_(-1, k.view(-1, 1), 1.0)
                .view(bsz * tsz, self.groups, -1)
            )
            hard_probs = torch.mean(hard_x.float(), dim=0)
            result["code_perplexity"] = torch.exp(
                -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
            ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(
                x
            )
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        # if print_option:
        #     idx = torch.randperm(x.size(-1))[0]
        #     print('idx:', idx)
        #     print('x:',x[idx])

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result
