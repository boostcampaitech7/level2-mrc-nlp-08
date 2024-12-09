from transformers import BertPreTrainedModel , RobertaPreTrainedModel , BigBirdPreTrainedModel
from transformers import BertModel, RobertaModel , BigBirdModel , AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.big_bird.modeling_big_bird import BigBirdForQuestionAnsweringModelOutput
from transformers.models.big_bird.modeling_big_bird import BigBirdIntermediate,BigBirdOutput

from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from typing import Optional, Tuple, Union


class Bert_CNN_Answering(BertPreTrainedModel):
    def __init__(self, config, model_path):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained(model_path, config=config, add_pooling_layer=False)

        # for name,param in self.bert.named_parameters():
        #     print(f"Parameter {name}: requires_grad={param.requires_grad}")

        for param in self.bert.parameters():
            param.requires_grad=True
            
        self.conv1 = nn.Conv1d(config.hidden_size, 128, kernel_size=3, stride=1, padding=1)  # Conv1d : (in_channels, out_channels, kernel_size) 500은 임의로 정함
        self.conv2 = nn.Conv1d(128, config.hidden_size, kernel_size=1, stride=1)  # Conv1d : (in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        
        # LayerNorm은 (batch_size, seq_len, hidden_size) 형식으로 입력을 받아야 하므로 Conv1d 이후 permute를 고려
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # start와 end 위치를 예측하는 Linear 레이어 정의
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(f"input_ids: {input_ids.size() if input_ids is not None else 'None'}")
        # print(f"attention_mask: {attention_mask.size() if attention_mask is not None else 'None'}")
        # print(f"token_type_ids: {token_type_ids.size() if token_type_ids is not None else 'None'}")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, # Roberta 모델 사용시 사용안함
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # Conv1d는 (batch_size, hidden_size, seq_len)
        conv_input = sequence_output.permute(0, 2, 1)

        # CNN 레이어를 5번 반복해서 통과시킴
        for _ in range(5):
            residual = conv_input
            conv1_output = self.conv1(conv_input)
            conv2_output = self.conv2(conv1_output)
            
            relu_output = self.relu(conv2_output)
            residual_output = relu_output + residual
            
            
            # LayerNorm#
            # 다시 (batch_size, seq_len, hidden_size)로 permute 
            conv_output = residual_output.permute(0, 2, 1)
            conv_input = self.layer_norm(conv_output)
            # layernorm #
            
            conv_input = conv_input.permute(0, 2, 1)
            

        # (batch_size, seq_len, hidden_size)
        norm_output = conv_input.permute(0, 2, 1)
        # QA 태스크를 위한 start, end logits 계산


        logits = self.qa_outputs(norm_output)
        
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # print('start_logits',start_logits)
            # print('start_positions',start_positions)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            print('start_loss',start_loss)
            print('end_loss',end_loss)
            print('total_loss',total_loss)


        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
             
# class Roberta_CNN_Answering(RobertaPreTrainedModel): 
#     def __init__(self, config, model_path):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         # self.bert = RobertaModel(config, add_pooling_layer=False)
#         self.bert = RobertaModel.from_pretrained(model_path, config=config, add_pooling_layer=False)

#         # for name,param in self.bert.named_parameters():
#         #     print(f"Parameter {name}: requires_grad={param.requires_grad}")

#         for param in self.bert.parameters():
#             param.requires_grad=True
            
#         self.conv1 = nn.Conv1d(config.hidden_size, 128, kernel_size=3, stride=1, padding=1)  # Conv1d : (in_channels, out_channels, kernel_size) 500은 임의로 정함
#         self.conv2 = nn.Conv1d(128, config.hidden_size, kernel_size=1, stride=1)  # Conv1d : (in_channels, out_channels, kernel_size)
#         self.relu = nn.ReLU()
        
#         # LayerNorm은 (batch_size, seq_len, hidden_size) 형식으로 입력을 받아야 하므로 Conv1d 이후 permute를 고려
#         self.layer_norm = nn.LayerNorm(config.hidden_size)

#         # start와 end 위치를 예측하는 Linear 레이어 정의
#         self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.init_weights()

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         start_positions: Optional[torch.Tensor] = None,
#         end_positions: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
#         r"""
#         start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # print(f"input_ids: {input_ids.size() if input_ids is not None else 'None'}")
#         # print(f"attention_mask: {attention_mask.size() if attention_mask is not None else 'None'}")
#         # print(f"token_type_ids: {token_type_ids.size() if token_type_ids is not None else 'None'}")

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             # token_type_ids=token_type_ids, # Roberta 모델 사용시 사용안함
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]
#         # Conv1d는 (batch_size, hidden_size, seq_len)
#         conv_input = sequence_output.permute(0, 2, 1)

#         # CNN 레이어를 5번 반복해서 통과시킴
#         for _ in range(5):
#             residual = conv_input
#             conv1_output = self.conv1(conv_input)
#             conv2_output = self.conv2(conv1_output)
            
#             relu_output = self.relu(conv2_output)
#             residual_output = relu_output + residual
            
            
#             # LayerNorm#
#             # 다시 (batch_size, seq_len, hidden_size)로 permute 
#             conv_output = residual_output.permute(0, 2, 1)
#             conv_input = self.layer_norm(conv_output)
#             # layernorm #
            
#             conv_input = conv_input.permute(0, 2, 1)
            

#         # (batch_size, seq_len, hidden_size)
#         norm_output = conv_input.permute(0, 2, 1)
#         # QA 태스크를 위한 start, end logits 계산


#         logits = self.qa_outputs(norm_output)
        
        
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions = start_positions.clamp(0, ignored_index)
#             end_positions = end_positions.clamp(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             # print('start_logits',start_logits)
#             # print('start_positions',start_positions)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#             # print('start_loss',start_loss)
#             # print('end_loss',end_loss)
#             print('total_loss',total_loss)


#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return QuestionAnsweringModelOutput(
#             loss=total_loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

class BigBird_QA_HEAD(nn.Module):
    """Head for question answering tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # CNN 레이어 추가 (Conv1d는 시퀀스 차원에 적용)
        self.conv1 = nn.Conv1d(config.hidden_size, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, config.hidden_size, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_output):
        hidden_states = self.dropout(encoder_output)
        
        cnn_input=hidden_states.permute(0,2,1)
        for _ in range(5):
            residual=cnn_input
            conv1_out=self.conv1(cnn_input)
            conv2_out=self.conv2(conv1_out)
            relu_output=self.relu(conv2_out)
            residual_output = relu_output + residual
            
            conv_output = residual_output.permute(0, 2, 1)
            conv_input = self.layer_norm(conv_output)
            # layernorm #
            
            conv_input = conv_input.permute(0, 2, 1)
        
        norm_output = conv_input.permute(0, 2, 1)
        
        
        hidden_states = self.intermediate(norm_output)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states


class BigBird_CNN_Answering(BigBirdPreTrainedModel):
    def __init__(self, config, model_path, add_pooling_layer=False):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id

        # self.bert = RobertaModel(config, add_pooling_layer=False)
        self.bert = BigBirdModel.from_pretrained(model_path, config=config, add_pooling_layer=add_pooling_layer)
        self.qa_classifier = BigBird_QA_HEAD(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_lengths: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BigBirdForQuestionAnsweringModelOutput, Tuple[torch.FloatTensor]]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seqlen = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)

        if question_lengths is None and input_ids is not None:
            # assuming input_ids format: <cls> <question> <sep> context <sep>
            question_lengths = torch.argmax(input_ids.eq(self.sep_token_id).int(), dim=-1) + 1
            question_lengths.unsqueeze_(1)

        logits_mask = None
        # print(question_lengths.shape) # [16,1]
        # print(seqlen.shape)
        
        # print(question_lengths)
        # print(seqlen) # 384 우리가 설정한 max_seq_len임
        
        if question_lengths is not None:
            # setting lengths logits to `-inf`
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = torch.ones(logits_mask.size(), dtype=int, device=logits_mask.device) - logits_mask
            logits_mask = logits_mask
            logits_mask[:, 0] = False
            logits_mask.unsqueeze_(2)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        ##
        
        logits = self.qa_classifier(sequence_output)

        if logits_mask is not None:
            # removing question tokens from the competition
            logits = logits - logits_mask * 1e6

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            print("total_loss",total_loss)
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BigBirdForQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 
    
    def prepare_question_mask(self,q_lengths: torch.Tensor, maxlen: int):
        # q_lengths -> (bz, 1)
        mask = torch.arange(0, maxlen).to(q_lengths.device)
        mask.unsqueeze_(0)  # -> (1, maxlen)
        mask = torch.where(mask < q_lengths, 1, 0)
        return mask