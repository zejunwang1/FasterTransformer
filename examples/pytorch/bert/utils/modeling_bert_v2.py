# Developed by Wang Zejun
import torch

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
        self.use_ext_encoder = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if self.use_ext_encoder:
            assert attention_mask.dim() == 2
            seq_lens = torch.sum(attention_mask, 1, dtype=torch.int32).to(device)
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
                )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if self.use_ext_encoder:
            encoder_outputs = self.encoder(embedding_output, seq_lens)
        else:
            head_mask = [None] * self.config.num_hidden_layers
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]

    def replace_encoder(self, encoder):
        self.use_ext_encoder = True
        self.encoder = encoder


