# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
                for key, value in json_config.items():
                    self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range


        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, 256 * 3)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, 256 * 3)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(256 * 3, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        if concat_type is None:
            concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)
        
        #print("concat_embeddings.shape:" + str(concat_embeddings.shape))
        #print("position_embeddings.shape:" + str(position_embeddings.shape))
        #print("token_type_embeddings.shape:" + str(token_type_embeddings.shape))
        
        embeddings = concat_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossSelfAttention(nn.Module):
    def __init__(self, config):
        super(CrossSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads   # 8
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 256 / 8 = 32
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 256

        self.query = nn.Linear(config.hidden_size, self.all_head_size)  # 256 * 256
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # Split self.K Q V into 12 heads for multi head splitting, and store the split K Q V in key_layer, query_layer, and value_layer
    # (0,1,2,3)-->(0,2,1,3):
    # (batch_size,seq_len,num_heads,attn_head_size)--->(batch_size,num_heads,seq_len,attn_head_size)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in CrossModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Calculated attn_probs have to multiply InputMatrix V (V is same as K)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [64, 120/180/240, 256]
        return context_layer


class CrossSelfOutput(nn.Module):
    def __init__(self, config):
        super(CrossSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = CrossSelfAttention(config)
        self.output = CrossSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class CrossIntermediate(nn.Module):
    def __init__(self, config):
        super(CrossIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CrossOutput(nn.Module):
    def __init__(self, config):
        super(CrossOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CrossLayer(nn.Module):
    def __init__(self, config):
        super(CrossLayer, self).__init__()
        self.attention = CrossAttention(config)
        self.intermediate = CrossIntermediate(config)
        self.output = CrossOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CrossEncoder(nn.Module):
    def __init__(self, config):
        super(CrossEncoder, self).__init__()
        layer = CrossLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])  # config.num_hidden_layers: 3

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:  # True
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers  # return last one


class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(CrossPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class CrossLMPredictionHead(nn.Module):
    def __init__(self, config, cross_model_embedding_weights):
        super(CrossLMPredictionHead, self).__init__()
        self.transform = CrossPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(cross_model_embedding_weights.size(1),
                                 cross_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = cross_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(cross_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class CrossOnlyMLMHead(nn.Module):
    def __init__(self, config, cross_model_embedding_weights):
        super(CrossOnlyMLMHead, self).__init__()
        self.predictions = CrossLMPredictionHead(config, cross_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class CrossOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(CrossOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class CrossPreTrainingHeads(nn.Module):
    def __init__(self, config, cross_model_embedding_weights):
        super(CrossPreTrainingHeads, self).__init__()
        self.predictions = CrossLMPredictionHead(config, cross_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score



# DAM and FDF
class CrossModel(PreTrainedModel):
    def __init__(self, config):
        super(CrossModel, self).__init__(config)
        self.embeddings = CrossEmbeddings(config)
        self.encoder1 = CrossEncoder(config)
        self.encoder2 = CrossEncoder(config)
        self.linear = nn.Linear(256 * 3, 256)
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def forward(self, private_text, private_visual, private_audio, common_text, common_visual, common_audio, output_all_encoded_layers=True):
        
        device = torch.device('cuda:0')
        batch_size = private_text.shape[0]
        
        # Domain Association Modeling DAM
        # calculate unified affinity matrix
        pc1 = torch.cat((private_text, common_text), 1)
        pc2 = torch.cat((private_visual, common_visual), 1)
        pc3 = torch.cat((private_audio, common_audio), 1)
        pc12 = torch.cat((pc1, pc2), 2)
        unified_affinity_matrix = torch.cat((pc12, pc3), 2)

        # calculate harmonic affinity matrix
        harmonic_private_feature = private_text + private_visual + private_audio
        harmonic_common_feature = common_text + common_visual + common_audio
        pc_matrix = torch.cat((harmonic_private_feature, harmonic_common_feature), 1)  # Rows Concat First
        cp_matrix = torch.cat((harmonic_common_feature, harmonic_private_feature), 1)
        pccp = torch.cat((pc_matrix, cp_matrix), 2)  # Then columns Concat
        cppc = torch.cat((cp_matrix, pc_matrix), 2)  # Then columns Concat, cppc have to Transposition, pccp X (cppc)T = HAM
        harmonic_affinity_matrix = torch.matmul(pccp, cppc.transpose(1, 2)).to(device)  # Square Matrix, HAM x UAM = FAM

        # calculate final affinity metrix
        final_affinity_metrix = torch.matmul(harmonic_affinity_matrix, unified_affinity_matrix).to(device)
        
        # Feature Dynamic Fusion Module
        # FDF
        concat_input = torch.cat((unified_affinity_matrix, final_affinity_metrix), 1)
        attention_mask = torch.ones(concat_input.size(0), concat_input.size(1)).to(device)
        
        uam_type = torch.ones(batch_size, 120, dtype=int)
        fam_type = torch.zeros(batch_size, 120, dtype=int)
        concat_type = torch.cat((uam_type, fam_type), 1).to(device)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # [64, 1, 1, 120] or [64, 1, 1, 18 0] or [64, 1, 1, 240], -0.
        
        embedding_output = self.embeddings(concat_input, concat_type)
        encoded_layers = self.encoder1(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        sigmoid_output = torch.sigmoid(sequence_output)
        padding = torch.ones(batch_size, 120, 256 * 3).to(device)  # before: 1
        padded_uam = torch.cat((unified_affinity_matrix, padding), 1)
        padded_fam = torch.cat((padding, final_affinity_metrix), 1)
        sig_uam = torch.mul(padded_uam, sigmoid_output)
        sig_fam = torch.mul(padded_fam, sigmoid_output)
        
        sig_input = sig_fam + sig_uam
        attention_mask = torch.ones(sig_input.size(0), sig_input.size(1), dtype=int).to(device)
        concat_type = torch.zeros(attention_mask.shape[0], attention_mask.shape[1], dtype=int).to(device)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # [64, 1, 1, 120] or [64, 1, 1, 18 0] or [64, 1, 1, 240], -0.
        
        embedding_output = self.embeddings(sig_input, concat_type)
        encoded_layers = self.encoder2(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_input = self.linear(sequence_output)
        #pooled_output = self.pooler(pooled_input)
        
        #print("pooled_output:" + str(pooled_output))
        
        return pooled_input
