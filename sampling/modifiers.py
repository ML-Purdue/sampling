from . import Distribution
import torch

class Modifier:
    def __init__(self):
        pass

    def __call__(self, dist: Distribution):
        raise NotImplementedError()


# Top K and Top P derived from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html

# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

class TopK(Modifier):
    def __init__(self, k: int):
        super().__init__()

        if k < 1 or not isinstance(k, int):
            raise ValueError(f'k must be a positive integer, got {k}')

        self.k = k

    def __call__(self, dist: Distribution):
        indices_to_remove = scores < torch.topk(dist.logits, self.k)[0][..., -1, None]
        dist.logits = scores.masked_fill(indices_to_remove, -float('Inf'))



class TopP(Modifier):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def __call__(self, dist: Distribution):
        if self.p >= 1.0:
            return dist

        sorted_logits, sorted_indices = torch.sort(dist.logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        dist.logits[indices_to_remove] = -float('Inf')

        return dist

