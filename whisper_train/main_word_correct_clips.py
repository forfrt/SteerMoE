"""
 * author Ruitao Feng
 * created on 10-07-2025
 * github: https://github.com/forfrt
"""

# codingse :utf8
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
# 设置可见显卡，0，1，2，3四张卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
# 接下来是你的训练代码

import transformers
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import Repository, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

# if is_fairscale_available():
#     dep_version_check("fairscale")
#     import fairscale
#     from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
#     from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
#     from fairscale.nn.wrap import auto_wrap
#     from fairscale.optim import OSS
#     from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
from transformers import Trainer
class Logged_Traniner(Trainer):
    # Overwrite the evaluation_loop function from Trainer
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        print("成功替换")
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            print(inputs['input_features'].dtype)
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
# transformers.Trainer = Logged_Traniner
from data_process import DatasetCommonVoiceHindi, DatasetCommonVoiceCN
from setting import model_path, tokenizer_path
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import WhisperProcessor, WhisperTokenizer
from modules import DataCollatorSpeechSeq2SeqWithPadding, ASREvaluator
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction, PredictionOutput
    from transformers.training_args import TrainingArguments


logger = logging.get_logger(__name__)
class Logged_Seq2SeqTrainer(Logged_Traniner):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config

    @staticmethod
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # GenerationConfig provided, nothing to do
        if isinstance(gen_config_arg, GenerationConfig):
            return deepcopy(gen_config_arg)

        # str or Path
        pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
        config_file_name = None

        # Figuring if it is path pointing to a file, pointing to a directory or else a model id or URL
        # This step is required in order to determine config_file_name
        if pretrained_model_name.is_file():
            config_file_name = pretrained_model_name.name
            pretrained_model_name = pretrained_model_name.parent
        # dir path
        elif pretrained_model_name.is_dir():
            pass
        # model id or URL
        else:
            pretrained_model_name = gen_config_arg

        gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)
        return gen_config

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self._gen_kwargs = gen_kwargs

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
def model_test():

    model_path = {
        "whisper_base": "./model/whisper_base/models--openai--whisper-base/snapshots/013fe3bf928b86dc6830b3dc7162d122562cab10",
        "whisper_large-v2": "./model/whisper_large_v2/696465c"
    }
    tokenizer_path = {
        "whisper_base": model_path["whisper_base"],
        "whisper_large-v2": model_path["whisper_large-v2"]
    }

    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path["whisper_large-v2"],
        tokenizer=tokenizer_path["whisper_large-v2"],
        chunk_length_s=30,
        device=device
    )

    import librosa
    import time
    # 加载音频文件
    audio_file = "../audio_files/陈果_2023年A股投资策略20221220.mp3"
    audio_input, sample_rate = librosa.load(audio_file, sr=16000)  # 确保采样率为 16000 Hz
    options = dict(language="Chinese")
    transcribe_options = dict(task="transcribe", **options)

    start = time.time()
    prediction = pipe(audio_input,
                      batch_size=1,
                      generate_kwargs={"task":"transcribe", "language":"chinese"},
                      return_timestamps=True)["chunks"]
    end = time.time()
    print("耗时：", end - start)

    # print(prediction)
    for item in prediction:
        print(item["timestamp"], item["text"])

from datasets import load_from_disk,DatasetDict,concatenate_datasets
import tqdm
def train(parquet_dirs:list,save_dir,resume_from_checkpoint=None,resume_from_model=None,output_dir='./model/whisper_large_v2_full_checkpoint',model_name='whisper_large-v2',language="chinese"):
# def train(output_dir='./model/whisper_base-cn', model_name='whisper_base'):
    # 加载数据集
    # # dataset = DatasetCommonVoiceHindi().load_processed_data()
    # # dataset = DatasetCommonVoiceCN().load_processed_data()
    # print('start loading dataset')
    # dataset = DatasetDict()
    # train_datasets = []
    # for folder in os.listdir('./tmp_scripts/aopeng_processed_in_batch_5k'):
    #     tmp = load_from_disk(os.path.join('./tmp_scripts/aopeng_processed_in_batch_5k', folder))
    #     train_datasets.append(tmp)
    # # combined_dataset = concatenate_datasets(train_datasets)
    # concat = train_datasets[0]
    # for tmp_dataset in tqdm.tqdm(train_datasets[1:]):
    #     concat = concatenate_datasets([concat, tmp_dataset])
    # dataset['train'] = concat
    # print(len(concat))
    # # dataset["test"] = load_from_disk('./tmp_scripts/tmp_data_test_processed')
    #
    # print("dataset loaded")
    # dataset = DatasetDict()
    # print('start loading dataset')
    # dataset['train'] = load_from_disk('./tmp_scripts/tmp_data_train_processed_5k_merged',keep_in_memory=True)
    # print("dataset loaded")
    print('start loading dataset')
    dataset = DatasetDict()
    train_datasets = []
    # parquet_dir = './tmp_scripts/aopeng_processed_in_batch_5k'
    # parquet_dir = './tmp_scripts/alphaengine_processed_in_batch_5k'
    # parquet_files = os.listdir(parquet_dir)
    # parquet_files = [file for file in parquet_files if int(file.split('_')[-1]) >3000]
    # print(len(parquet_files))
    for folder in tqdm.tqdm(parquet_dirs):
        tmp = load_from_disk(folder)
        train_datasets.append(tmp)
    # combined_dataset = concatenate_datasets(train_datasets)
    concat = train_datasets[0]
    for tmp_dataset in tqdm.tqdm(train_datasets[1:]):
        concat = concatenate_datasets([concat, tmp_dataset])
    dataset['train'] = concat
    print(len(concat))

    # 加载特征提取器和分词器
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                 language=language, task="transcribe",
                                                 cache_dir="./model")
    feature_extractor=processor.feature_extractor
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                 language=language, task="transcribe")
    print('processor and tokenizer loaded')
    # time.sleep(20)
    # 加载评估器
    asr_metrics = ASREvaluator(tokenizer=tokenizer)

    # 加载模型 checkpoint
    from transformers import WhisperForConditionalGeneration

    if not resume_from_model:
        model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                                local_files_only=True)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=resume_from_model,
                                                            local_files_only=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05

    print(f"model.config: {model.config}")
    print("model loaded")

    max_label_length = model.config.max_target_positions
    print(f"max_label_length: {max_label_length}")

    MAX_DURATION_IN_SECONDS = 30.0
    max_input_length = MAX_DURATION_IN_SECONDS * 16000

    def filter_inputs(input_length):
        """Filter inputs with zero input length or longer than 30s""" 
        return 0 < input_length < max_input_length

    def filter_labels(labels):
        """Filter label sequences longer than max length (448)"""
        return len(labels) < max_label_length

    print(f"dataset before filter: {dataset}")
    # filter by audio length
    dataset = dataset.filter(filter_inputs, input_columns=["input_length"])
    # filter by label length
    dataset = dataset.filter(filter_labels, input_columns=["labels"])
    print(f"dataset after filter: {dataset}")


    # 训练模型
    training_args = Seq2SeqTrainingArguments(
        resume_from_checkpoint=resume_from_checkpoint,
        output_dir=output_dir,  # change to a repo name of your choice
        # per_device_train_batch_size=16, # it will take about 23gb memory
        # per_device_train_batch_size=48, # it will take about 39gb memory
        # per_device_train_batch_size=64, # it will take about 51gb memory
        per_device_train_batch_size=96, # it will take about 67gb memory
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        # max_steps=50000,
        num_train_epochs=1,
        gradient_checkpointing=True,
        fp16=True,
        weight_decay=0.01,
        evaluation_strategy="no",
        # per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="steps",
        save_steps=1000,
        # eval_steps=2000,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        push_to_hub=False,
        deepspeed="./ds_configs/stage2.json",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=asr_metrics.compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # 保存模型
    trainer.save_model(save_dir)

def train_in_ddp(output_dir='./model/whisper_large_v2_full_checkpoint',model_name='whisper_large-v2'):
# def train(output_dir='./model/whisper_base-cn', model_name='whisper_base'):
    # 加载数据集
    # # dataset = DatasetCommonVoiceHindi().load_processed_data()
    # # dataset = DatasetCommonVoiceCN().load_processed_data()
    # print('start loading dataset')
    # dataset = DatasetDict()
    # train_datasets = []
    # for folder in os.listdir('./tmp_scripts/aopeng_processed_in_batch_5k'):
    #     tmp = load_from_disk(os.path.join('./tmp_scripts/aopeng_processed_in_batch_5k', folder))
    #     train_datasets.append(tmp)
    # # combined_dataset = concatenate_datasets(train_datasets)
    # concat = train_datasets[0]
    # for tmp_dataset in tqdm.tqdm(train_datasets[1:]):
    #     concat = concatenate_datasets([concat, tmp_dataset])
    # dataset['train'] = concat
    # print(len(concat))
    # # dataset["test"] = load_from_disk('./tmp_scripts/tmp_data_test_processed')
    #
    # print("dataset loaded")
    # dataset = DatasetDict()
    # print('start loading dataset')
    # dataset['train'] = load_from_disk('./tmp_scripts/tmp_data_train_processed_5k_merged')
    # print("dataset loaded")
    print('start loading dataset')
    dataset = DatasetDict()
    train_datasets = []
    for folder in tqdm.tqdm(os.listdir('./tmp_scripts/aopeng_processed_in_batch_5k')):
        tmp = load_from_disk(os.path.join('./tmp_scripts/aopeng_processed_in_batch_5k', folder))
        train_datasets.append(tmp)
    # combined_dataset = concatenate_datasets(train_datasets)
    concat = train_datasets[0]
    for tmp_dataset in tqdm.tqdm(train_datasets[1:]):
        concat = concatenate_datasets([concat, tmp_dataset])
    dataset['train'] = concat
    print(len(concat))

    print("dataset loaded")

    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend="nccl")

    # 加载特征提取器和分词器
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                 language="chinese", task="transcribe",
                                                 cache_dir="./model")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                 language="chinese", task="transcribe")
    print('processor and tokenizer loaded')
    # time.sleep(20)
    # 加载评估器
    asr_metrics = ASREvaluator(tokenizer=tokenizer)

    # 加载模型 checkpoint
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                            local_files_only=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = model.to(torch.device("cuda"))
    model = DDP(model)
    print("model loaded")

    # 添加环境变量
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    train_sampler = DistributedSampler(dataset["train"])
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # 训练模型
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        # max_steps=50000,
        num_train_epochs=2,
        gradient_checkpointing=True,
        fp16=True,
        weight_decay=0.01,
        evaluation_strategy="no",
        # per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="steps",
        save_steps=1000,
        # eval_steps=2000,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        push_to_hub=False,
        # deepspeed="../ds_configs/stage2.json",
        deepspeed=None,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=asr_metrics.compute_metrics,
        tokenizer=processor.feature_extractor,
        train_sampler=train_sampler,  # 添加分布式采样器
    )

    trainer.train()

    # 保存模型
    trainer.save_model("./model/whisper_large_v2_full_checkpoint/model_{}".format(training_args.max_steps))

class CustomDatasetIterator:
    def __init__(self, dataset_paths, batch_size, total_size):
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.total_size = total_size
        self.current_dataset_index = 0
        self.current_dataset = load_from_disk(dataset_paths[0])
        self.current_dataset_iter = iter(self.current_dataset)
    def __len__(self):
        return self.total_size

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        while len(batch) < self.batch_size:
            try:
                item = next(self.current_dataset_iter)
                batch.append(item)
            except StopIteration:
                self.current_dataset_index += 1
                if self.current_dataset_index >= len(self.dataset_paths):
                    raise StopIteration
                self.current_dataset = load_from_disk(self.dataset_paths[self.current_dataset_index])
                self.current_dataset_iter = iter(self.current_dataset)
        return batch
def train_in_batch(output_dir='./model/whisper_large_v2_full_checkpoint',model_name='whisper_large-v2'):
    # 数据集路径列表
    dataset_paths = list(os.listdir('./tmp_scripts/aopeng_processed_in_batch_5k'))
    dataset_paths = ['./tmp_scripts/aopeng_processed_in_batch_5k/{}'.format(path) for path in dataset_paths]
    total_size = 1604566  # 假设的总数据条目数
    batch_size = 32  # 每个批次的大小

    # 创建自定义数据集迭代器
    dataset_iterator = CustomDatasetIterator(dataset_paths, batch_size, total_size)

    print("dataset loaded")
    # 加载特征提取器和分词器
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                 language="chinese", task="transcribe",
                                                 cache_dir="./model")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                 language="chinese", task="transcribe")
    print('processor and tokenizer loaded')
    # time.sleep(20)
    # 加载评估器
    asr_metrics = ASREvaluator(tokenizer=tokenizer)

    # 加载模型 checkpoint
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path[model_name],
                                                            local_files_only=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("model loaded")

    # 添加环境变量
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 训练模型
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        # max_steps=50000,
        num_train_epochs=2,
        gradient_checkpointing=True,
        fp16=True,
        weight_decay=0.01,
        evaluation_strategy="no",
        # per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="steps",
        save_steps=1000,
        # eval_steps=2000,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        push_to_hub=False,
        deepspeed="../ds_configs/stage2.json",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_iterator,
        # eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=asr_metrics.compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # 保存模型
    trainer.save_model("./model/whisper_large_v2_full_checkpoint_aopeng5k_alpha3k/model_{}".format(training_args.max_steps))

def train_lora():

    # 加载数据集
    # dataset = DatasetCommonVoiceHindi().load_processed_data()
    # dataset = DatasetCommonVoiceCN().load_processed_data()
    dataset = DatasetDict()
    train_datasets = []
    for folder in os.listdir('./tmp_scripts/aopeng_processed_in_batch'):
        tmp = load_from_disk(os.path.join('./tmp_scripts/aopeng_processed_in_batch', folder))
        train_datasets.append(tmp)
    combined_dataset = concatenate_datasets(train_datasets)
    dataset['train'] = combined_dataset
    dataset["test"] = load_from_disk('./tmp_scripts/tmp_data_test_processed')
    print("dataset loaded")

    # 加载特征提取器和分词器
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                                 language="chinese", task="transcribe",
                                                 cache_dir="./model")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                                 language="chinese", task="transcribe")

    # 加载评估器
    asr_metrics = ASREvaluator(tokenizer=tokenizer)

    # 加载模型 checkpoint
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                                            local_files_only=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("model loaded")

    # 配置 Lora
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 添加环境变量
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # 训练模型
    training_args = Seq2SeqTrainingArguments(
        output_dir="temp",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        weight_decay=0.01,
        fp16=True,
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=3,
        save_strategy = "steps",
        save_steps = 2000,
        eval_steps = 2000,
        report_to = ["tensorboard"],
        evaluation_strategy="steps",
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        per_device_eval_batch_size=4,
        generation_max_length=225,
        logging_steps=25,
        remove_unused_columns=False,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
        push_to_hub=False,
        deepspeed="../ds_configs/stage2.json",
    )

    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

            return control

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=asr_metrics.compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback]
    )

    trainer.train()

    # 保存模型
    # trainer.save_model("./model/whisper_base-cn/model_{}".format(training_args.max_steps))


def train_lora_in_8bit():

    # 加载数据集
    # dataset = DatasetCommonVoiceHindi().load_processed_data()
    # dataset = DatasetCommonVoiceCN().load_processed_data()
    dataset = DatasetDict()
    train_datasets = []
    for folder in os.listdir('./tmp_scripts/aopeng_processed_in_batch'):
        tmp = load_from_disk(os.path.join('./tmp_scripts/aopeng_processed_in_batch', folder))
        train_datasets.append(tmp)
    combined_dataset = concatenate_datasets(train_datasets)
    dataset['train'] = combined_dataset
    # dataset["test"] = load_from_disk('./tmp_scripts/tmp_data_test_processed')
    print("dataset loaded")

    # 加载特征提取器和分词器
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                                 language="chinese", task="transcribe",
                                                 cache_dir="./model")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                                 language="chinese", task="transcribe")

    # 加载评估器
    asr_metrics = ASREvaluator(tokenizer=tokenizer)

    # 加载模型 checkpoint in 8bit
    from transformers import WhisperForConditionalGeneration
    from peft import prepare_model_for_int8_training

    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path["whisper_large-v2"],
                                                            local_files_only=True,
                                                            load_in_8bit=True,
                                                            device_map="auto")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("model loaded")

    model = prepare_model_for_int8_training(model)

    # 配置 Lora
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 添加环境变量
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # 训练模型
    training_args = Seq2SeqTrainingArguments(
        output_dir="./model/whisper_large_v2_checkpoint",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        weight_decay=0.01,
        fp16=True,
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=3,
        save_strategy = "steps",
        save_steps = 2000,
        # eval_steps = 2000,
        report_to = ["tensorboard"],
        evaluation_strategy="no",
        predict_with_generate=True,
        load_best_model_at_end=False,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        # per_device_eval_batch_size=4,
        generation_max_length=225,
        logging_steps=25,
        remove_unused_columns=False,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
        push_to_hub=False,
        deepspeed="../ds_configs/stage2.json",
    )

    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

            return control

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=asr_metrics.compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback]
    )

    trainer.train()

    # 保存模型
    # trainer.save_model("./model/whisper_base-cn/model_{}".format(training_args.max_steps))

import setting
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_tag', required=True, help='Batch ID')
    parser.add_argument('--last_saved', required=True, help='last saved checkpoint')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    last_saved=args.last_saved
    print(f"last saved checkpoint: {last_saved}")

    if args.batch_tag:
        print(f"batch_tag from arugments: {args.batch_tag}")
        batch_tag=args.batch_tag
    else:
        batch_tag = f'{setting.SOURCE_TAG}_{setting.BATCH_TAG}'
        print(f"batch_tag from setting.json: {batch_tag}")

    batch_tags=[batch_tag]

    for id, batch_tag in enumerate(batch_tags):

        print(f"Batch IDs to process: {batch_tag}")

        processed_parquet_dir = f'/root/autodl-nas/ruitao/data/train/processed_parquet/{batch_tag}_processed_in_batch'
        # processed_parquet_dir = f'/root/autodl-tmp/ruitao/whisper_test/data/train/processed_parquet/{batch_tag}_processed_in_batch'
        parquet_folder=processed_parquet_dir

        print(f"parquet_folder: {parquet_folder}")

        if not os.path.exists(parquet_folder):
            raise FileNotFoundError(f"parquet_folder: {parquet_folder} not found")
        else:
            size = os.popen(f'du -s {parquet_folder}').read().split('\t')[0]
            print(f"Size of {parquet_folder}: {size} bytes")
            # @ruitao For TEST 
            # if int(size) < 60 * 1024 * 1024:
            if int(size) < 1 * 1024 * 1024:
                raise Exception(f"parquet_folder: {parquet_folder} size is less than 1G")

        resume_from_model = f'/root/autodl-tmp/ruitao/whisper_test/model/whisper_large_v2_{last_saved}/model_end'
        print(f"Resume from model: {resume_from_model}")
        
        save_dir = f'/root/autodl-tmp/ruitao/whisper_test/model/whisper_large_v2_{batch_tag}/model_end'
        output_dir = f'/root/autodl-tmp/ruitao/whisper_test/model/whisper_large_v2_{batch_tag}'

        print(f"save_dir: {save_dir}")
        print(f"output_dir: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        print(f"Starting training for batch_id: {batch_tag}")
        train(
            parquet_dirs=[os.path.join(parquet_folder, i) for i in os.listdir(parquet_folder)],
            resume_from_model=resume_from_model,
            save_dir=save_dir,
            output_dir=output_dir
        )
        
        print(f"Training completed for batch_id: {batch_tag}")

        # Let the model be saved completely before loaded 
        time.sleep(600)

        # if os.360.exists(parquet_folder):
        #     shutil.rmtree(parquet_folder)
        #     print(f"Deleted parquet_folder: {parquet_folder}")

        # 标记完成
        # with open(f"../../status_folder/train_model_{batch_tag}.done", "w") as f:
        #     f.write("done")
        print(f"Marked completion for batch_id: {batch_tag}")


