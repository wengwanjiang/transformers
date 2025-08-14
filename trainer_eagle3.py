from transformers import Trainer, TrainingArguments
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import Dataset

from .generation.configuration_utils import GenerationConfig
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .integrations.fsdp import is_fsdp_managed_module
from .trainer import Trainer
from .utils import is_datasets_available, logging
from .utils.deprecation import deprecate_kwarg


if is_datasets_available():
    import datasets

if TYPE_CHECKING:
    from torch.utils.data import IterableDataset

    from .data.data_collator import DataCollator
    from .feature_extraction_utils import FeatureExtractionMixin
    from .image_processing_utils import BaseImageProcessor
    from .modeling_utils import PreTrainedModel
    from .processing_utils import ProcessorMixin
    from .tokenization_utils_base import PreTrainedTokenizerBase
    from .trainer_callback import TrainerCallback
    from .trainer_utils import EvalPrediction, PredictionOutput
    from .training_args import TrainingArguments
# def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#     """
#     model (nn.Module) — The model to compute the loss for.
#     inputs (dict[str, Union[torch.Tensor, Any]]) — The input data for the model.
#     return_outputs (bool, optional, defaults to False) — Whether to return the model outputs along with the loss.
#     num_items_in_batch (Optional[torch.Tensor], optional) — The number of items in the batch. If num_items_in_batch is not passed,

#     return loss for eagle3
#     """
#     # Draft Model 前向传播
#     input_ids = inputs.get('input_ids', None)
#     attention_mask = inputs.get('attention_mask', None)
#     loss_mask = inputs.get('loss_mask', None)
#     past_key_values = inputs.get('past_key_values', None)
#     position_ids = inputs.get('position_ids', None)
#     plosses, vlosses, acces = model.compute_loss(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask,\
#                                                  past_key_values=past_key_values, position_ids=position_ids)
#     return plosses, vlosses, acces
    # """
    # Args:
    #         input_ids: (batch, seq_len)
    #         attention_mask: (batch, seq_len)
    #         loss_mask: (batch, seq_len)
    #         past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
    #         position_ids: (batch, seq_len)
    # """

    # # Step 1: prepare data with the target model 
    # hidden_states, target, loss_mask, input_ids = model._prepare_data(input_ids, attention_mask, loss_mask)

    # # basic info
    # batch_size, seq_length, _ = hidden_states.shape
    # seq_length_with_past = seq_length
    # past_key_values_length = 0

    # # Step 2: project the concatenated hidden states to the target hidden size
    # hidden_states = model.project_hidden_states(hidden_states)
    # # Step 3: process kv cache, position ids and position ids
    # if past_key_values is not None:
    #     past_key_values_length = past_key_values[0][0].shape[2]
    #     seq_length_with_past = seq_length_with_past + past_key_values_length
    # if position_ids is None:
    #     device = hidden_states.device
    #     position_ids = torch.arange(
    #         past_key_values_length,
    #         seq_length + past_key_values_length,
    #         dtype=torch.long,
    #         device=device,
    #     )
    #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    # else:
    #     position_ids = position_ids.view(-1, seq_length).long()

    # # Step 4: handle attention mask
    # if attention_mask is None:
    #     attention_mask = torch.ones(
    #         (batch_size, seq_length_with_past),
    #         dtype=torch.bool,
    #         device=hidden_states.device,
    #     )
    # attention_mask = self.draft_model.prepare_decoder_attention_mask(
    #     attention_mask=attention_mask,
    #     hidden_states=hidden_states,
    #     batch_size=batch_size,
    #     seq_length=seq_length,
    #     past_key_values_length=past_key_values_length,
    # )

    # # Step 5: run TTT
    # plosses = []
    # vlosses = []
    # acces = []
    # cache_hidden = [[], []]

    # for idx in range(self.length):
    #     is_last = idx == self.length - 1

    #     # Step 5.1: embed the input ids
    #     inputs_embeds = self.draft_model.embed_input_ids(input_ids)
    #     inputs_embeds = inputs_embeds.to(hidden_states.dtype)

    #     # Step 5.2: run the draft model backbone
    #     hidden_states_out = self.draft_model.backbone(
    #         input_embeds=inputs_embeds,
    #         hidden_states=hidden_states,
    #         cache_hidden=cache_hidden,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         use_cache=True,
    #     )

    #     # Step 5.3: handle vocab size
    #     with torch.no_grad():
    #         target_head = target
    #         target_max_token = target_head.argmax(-1)
    #         target_mask = self.draft_model.t2d[target_max_token]
    #         target_mask = target_mask[..., None].int()
    #         position_mask = target_mask * loss_mask
    #         target_head = target_head[..., self.draft_model.t2d]
    #         target_head = target_head.float()
    #         target_p = nn.Softmax(dim=2)(target_head)
    #         target_p = target_p.detach()

    #     # update hidden states for next step
    #     hidden_states = hidden_states_out

    #     # Step 5.4: get logits
    #     logits = self.draft_model.compute_logits(hidden_states)
    #     logits = logits.float()

    #     # Step 5.5: calculate loss
    #     out_logp = nn.LogSoftmax(dim=2)(logits)
    #     plogp = target_p * out_logp
    #     loss = -torch.sum(position_mask * plogp, 2).mean()

    #     # Step 5.6: record metrics
    #     plosses.append(loss)
    #     with torch.no_grad():
    #         acces.append(
    #             (
    #                 (logits.argmax(-1) == target_p.argmax(-1))
    #                 * position_mask.squeeze(-1)
    #             )
    #             .sum()
    #             .item()
    #             / (loss_mask.sum().item() + 1e-6)
    #         )

    #     if not is_last:
    #         # Step 5.7: we need to update the loss mask
    #         input_ids = padding(input_ids, left=False)
    #         target = padding(target, left=False)
    #         loss_mask = padding(loss_mask, left=False)
    #         ind = torch.arange(seq_length, device=attention_mask.device)
    #         ind0 = ind[idx:]
    #         ind1 = ind[: seq_length - idx]
    #         attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min
    # return plosses, vlosses, acces


class Eagle3Trainer(Trainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Union[Dataset, "IterableDataset", "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union["PreTrainedTokenizerBase", "BaseImageProcessor", "FeatureExtractionMixin", "ProcessorMixin"]
        ] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], dict]] = None,
        callbacks: Optional[list["TrainerCallback"]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.model = model
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        model (nn.Module) — The model to compute the loss for.
        inputs (dict[str, Union[torch.Tensor, Any]]) — The input data for the model.
        return_outputs (bool, optional, defaults to False) — Whether to return the model outputs along with the loss.
        num_items_in_batch (Optional[torch.Tensor], optional) — The number of items in the batch. If num_items_in_batch is not passed,

        return loss for eagle3
        """

        input_ids = inputs.get('input_ids', None)
        attention_mask = inputs.get('attention_mask', None)
        loss_mask = inputs.get('loss_mask', None)
        past_key_values = inputs.get('past_key_values', None)
        position_ids = inputs.get('position_ids', None)
        plosses, vlosses, acces = model(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask,\
                                                    past_key_values=past_key_values, position_ids=position_ids)
        return plosses, vlosses, acces
    

#新人文档  大家遇到其他的问题可以也更新在上面   http://cf.myhexin.com/pages/viewpage.action?pageId=1320934321