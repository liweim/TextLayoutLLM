import transformers
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers.utils import (
    logging,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    is_peft_available,
    is_sagemaker_mp_enabled,

)
from typing import Optional
import os
import torch

logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftModel

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Trainer(transformers.Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            compute_loss=None,
    ):
        super(Trainer, self).__init__(
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
        self.loss_func = compute_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        return self.loss_func(model, inputs, self.args, return_outputs)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        torch.cuda.empty_cache()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        return (loss, None, None)

    def _load_from_peft_checkpoint(self, resume_from_checkpoint, model):
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        if not any(
                os.path.isfile(f) for f in [adapter_weights_file, adapter_safe_weights_file]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")
        print(f'resume from checkpoint: {resume_from_checkpoint}')
        print('!' * 100)
        # Load adapters following PR # 24096
        if is_peft_available() and isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint) or os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter)
                    # Load_adapter has no return value present, modify it when appropriate.
                    from torch.nn.modules.module import _IncompatibleKeys

                    load_result = _IncompatibleKeys([], [])
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"using `TrainerCallback` to save {ADAPTER_WEIGHTS_NAME} in corresponding folders, "
                        "here are some examples https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if is_peft_available() and isinstance(model, PeftModel):
            # Try to load adapters before trying to load a torch model
            try:
                return self._load_from_peft_checkpoint(resume_from_checkpoint, model=model)
            except:
                return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            # If it is not a PeftModel, use the original _load_from_checkpoint
        else:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)


class LoRATrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
