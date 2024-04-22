import torch
import torch.nn as nn


class Loss(object):
    def __call__(self, model, inputs, training_args, return_outputs=False):
        raise NotImplemented


class TargetLMLoss(Loss):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss
