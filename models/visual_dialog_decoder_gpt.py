import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils import logging
from transformers import BertGenerationDecoder, BertGenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import GPT2LMHeadModel
from models.vilbert_dialog import BertEmbeddingsDialog


#TODO create GPT decoder
class GPTVisualDialogDecoder(nn.Module):
    def __init__(self, params):
        super(GPTVisualDialogDecoder, self).__init__()
        self.params = params
        # self.config = GPT2Config()
        # print(f'{self.config= }')
        # input()
        # self.config.__dict__['cur_device'] = params["gpu_ids"][0]

        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", add_cross_attention=True)
        # self.decoder.config.add_cross_attention=True
        self.config = self.decoder.config
        self.config.pad_token_id = 0

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        loss_reduction=True
    ):

        # input_ids: [b, seq_len] -- torch.LongTensor
        # attention_mask: [b, seq_len] (0=masked or 1=not masked) -- torch.FloatTensor
        # encoder_hidden_states: [b, seq_len, num_hid] -- torch.FloatTensor
        # encoder_attention_mask: [b, seq_len] (0=masked or 1=not masked) -- torch.FloatTensor
        # labels: [b, seq_len] (ignore_index=0)

        if (labels is None) and (input_ids is not None):
            # shifting tokens to left for labels
            labels = input_ids.new_zeros(input_ids.shape)
            labels[:, :-1] = input_ids[:, 1:].clone()
            input_ids.masked_fill_(input_ids == self.config.eos_token_id, self.config.pad_token_id)

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        loss = None
        if 'train' in self.params['mode'] or 'eval' in self.params['mode']:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            if loss_reduction:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )



