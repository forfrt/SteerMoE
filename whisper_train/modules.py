"""
 * author Ruitao Feng
 * created on 16-07-2025
 * github: https://github.com/forfrt
"""

# coding:utf8
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from evaluate import load

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# TODO: 增加文本 normalizer
class ASREvaluator(object):

    def __init__(self, tokenizer):
        self.wer_metric = load("../evaluation/wer.py")
        # self.wer_metric = load("../../evaluation/wer.py")
        self.tokenizer = tokenizer

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def compute_metrics_by_str(self, pred_str, label_str):
        pred_str = " ".join(list(pred_str))
        label_str = " ".join(list(label_str))
        wer = 100 * self.wer_metric.compute(predictions=[pred_str], references=[label_str])
        return {"wer": wer}


if __name__ == '__main__':

    from evaluate import load

    wer_metric = load("./evaluation/wer.py")
    a = "10月率降到17年以来的 新级点"
    b = "失业率降到十七年来的新低点"
    a = " ".join(list(a))
    b = " ".join(list(b))
    print(a, b)
    rest = wer_metric.compute(predictions=[a], references=[b])
    print(rest)
    rest = wer_metric.compute(predictions=["good nidfht moon"], references=["good night moon"])
    print(rest)
