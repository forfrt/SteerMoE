# coding:utf8
import os

import torch
from datasets import Audio
from datasets import load_from_disk,concatenate_datasets
from setting import model_path, tokenizer_path
from datasets import DatasetDict, load_dataset

class DatasetCommonVoiceHindi(object):
    data_path = "./data/common_voice_11_0_hindi_processed"
    raw_data_path = "./data/common_voice_11_0"

    def __init__(self):
        pass

    def load_raw_data(self):
        """读取音频原始强度特征的数据"""
        common_voice = load_from_disk(self.raw_data_path)
        return common_voice

    def load_processed_data(self):
        """读取处理后的数据，其中音频原始强度特征转为 Mel 特征"""
        common_voice = load_from_disk(self.data_path)
        return common_voice


class DatasetCommonVoiceCN(DatasetCommonVoiceHindi):
    data_path = "./data/common_voice_zh_CN/dataset_processed"
    raw_data_path = "./data/common_voice_zh_CN/raw_dataset"

    def load_raw_data(self):

        common_voice = DatasetDict()
        common_voice["test"] = load_dataset(path=self.raw_data_path, split="test")
        common_voice["train"] = load_dataset(path=self.raw_data_path, split="train")

        common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes",
                                                    "gender", "locale", "path", "segment", "up_votes"])
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

        return common_voice


class DatasetFinAudioCN(DatasetCommonVoiceHindi):
    data_path = "./data/common_voice_zh_CN/dataset_processed"
    raw_test_data_path = "./data/finaudio_zh_test/raw_dataset/test"
    raw_train_data_path = "./data/finaudio_zh/raw_dataset/train"

    def load_raw_data(self):

        dataset = DatasetDict()
        data_dir= "tmp_scripts/tmp_data_test"
        tmp_datasets = []
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            tmp_dataset = load_dataset("parquet", data_files=file_path, split='train', cache_dir='.cache')
            tmp_dataset = tmp_dataset.add_column('file_name', [file]*len(tmp_dataset))
            tmp_datasets.append(tmp_dataset)
            print(tmp_dataset.column_names)
        dataset["test"] = concatenate_datasets(tmp_datasets)
        # dataset["test"] = load_dataset("parquet", data_dir="tmp_scripts/tmp_data_test", split='train', cache_dir='.cache')
        # dataset["train"] = load_dataset(path=self.raw_train_data_path, split="train")

        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        return dataset


class HFDataProcessorPipeline(object):
    """ 用于对于从 HuggingFace Datasets 中加载的数据进行预处理的管道
        初始数据是直接从 HuggingFace 仓库上下载 .parquet 格式
        最终数据是经过预处理后的 .arrow 格式存于本地硬盘
    """

    def __init__(self, feature_extractor, tokenizer):
         self.feature_extractor = feature_extractor
         self.tokenizer = tokenizer

    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = \
        self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # compute the input length
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch


def common_voice_hi_process():

    from datasets import load_from_disk

    # Init feature extractor and tokenizer
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path["whisper_base"])
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_base"],
                                                 language="Hindi", task="transcribe")
    data_pipe = HFDataProcessorPipeline(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load the raw dataset
    common_voice = load_from_disk("./data/common_voice_11_0")
    # common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes",
    #                                         "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    print(common_voice["train"][0])

    # 保留 common_voice 中的前100个样本
    # common_voice["train"] = common_voice["train"].select(range(100))
    # common_voice["test"] = common_voice["test"].select(range(100))
    # print(common_voice)

    # features extraction
    common_voice = common_voice.map(data_pipe.prepare_dataset,
                                    remove_columns=common_voice.column_names["train"], num_proc=1)
    # print(common_voice)
    # print(common_voice["input_features"][0])

    # save data to disk
    common_voice.save_to_disk("./data/common_voice_11_0_hindi_processed")


def common_voice_cn_process():
    """ 预处理 common voice 中文语音数据集 """

    from datasets import load_dataset
    from datasets import DatasetDict

    # Init feature extractor and tokenizer
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path["whisper_base"])
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_base"],
                                                 language="Hindi", task="transcribe")
    data_pipe = HFDataProcessorPipeline(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load the raw dataset
    common_voice = DatasetDict()
    common_voice["test"] = load_dataset(path="./data/common_voice_zh_CN/raw_dataset", split="test")
    # common_voice["train"] = load_dataset(path="./data/common_voice_zh_CN/raw_dataset", split="train+validation")

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes",
                                                "gender", "locale", "path", "segment", "up_votes"])
    # 迭代打印 common_voice 前10个样本
    for i in range(10):
        print(common_voice["test"][i])
        print(common_voice["test"][i]["audio"]["array"].shape)

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    print(common_voice["test"][0])

    # 保留 common_voice 中的前100个样本
    # common_voice["train"] = common_voice["train"].select(range(100))
    # common_voice["test"] = common_voice["test"].select(range(100))
    # print(common_voice)

    # features extraction
    common_voice = common_voice.map(data_pipe.prepare_dataset,
                                    remove_columns=common_voice.column_names["test"], num_proc=1)
    # print(common_voice)
    # print(common_voice["input_features"][0])

    # save data to disk
    common_voice.save_to_disk("./data/common_voice_zh_CN/dataset_processed")


def raw_dataset_cn_process():
    """ 预处理 common voice 中文语音数据集 """

    from datasets import load_dataset
    from datasets import DatasetDict

    # Init feature extractor and tokenizer
    # from transformers import WhisperFeatureExtractor, WhisperTokenizer
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path["whisper_base"])
    # tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_base"],
    #                                              language="Hindi", task="transcribe")
    # data_pipe = HFDataProcessorPipeline(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load the raw dataset
    dataset = DatasetDict()
    dataset["test"] = load_dataset(path="./data/finaudio_zh/raw_dataset", split="test")
    # common_voice["train"] = load_dataset(path="./data/common_voice_zh_CN/raw_dataset", split="train+validation")

    # common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes",
    #                                             "gender", "locale", "path", "segment", "up_votes"])
    # 迭代打印 common_voice 前10个样本
    print(dataset)
    for i in range(100):
        print(dataset["test"][i]["sentence"])
        print(dataset["test"][i]["audio"]["sampling_rate"])

    # common_voice = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # print(common_voice["test"][0])

    # 保留 common_voice 中的前100个样本
    # common_voice["train"] = common_voice["train"].select(range(100))
    # common_voice["test"] = common_voice["test"].select(range(100))
    # print(common_voice)

    # features extraction
    # common_voice = common_voice.map(data_pipe.prepare_dataset,
    #                                 remove_columns=common_voice.column_names["test"], num_proc=1)
    # print(common_voice)
    # print(common_voice["input_features"][0])

    # save data to disk
    # common_voice.save_to_disk("./data/common_voice_zh_CN/dataset_processed")


def load_data_test():

    from datasets import load_from_disk

    # Load the dataset
    # common_voice = DatasetDict()
    # common_voice["train"] = load_dataset(path="mozilla-foundation/common_voice_11_0",
    #                                      name="hi", split="train+validation",
    #                                      cache_dir="./data")
    # print("common_voice: ")
    # common_voice["test"] = load_dataset(path="mozilla-foundation/common_voice_11_0",
    #                                     name="hi", split="test",
    #                                     cache_dir="./data")
    # common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes",
    #                                             "gender", "locale", "path", "segment", "up_votes"])
    # common_voice.save_to_disk("./data/common_voice")

    # 保留 common_voice 中的前100个样本
    # common_voice["train"] = common_voice["train"].select(range(100))
    # common_voice["test"] = common_voice["test"].select(range(100))
    # print(common_voice)

    common_voice = load_from_disk("./data/common_voice_11_0")
    print(common_voice["train"][0])

    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path["whisper_base"])
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=model_path["whisper_base"],
                                                 language="Hindi", task="transcribe")

    input_str = common_voice["train"][0]["sentence"]
    labels = tokenizer(input_str).input_ids
    print("labels: ", labels)
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    print("decoded_with_special: ", decoded_with_special)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
    print("decoded_str: ", decoded_str)

    return common_voice


if __name__ == '__main__':

    # common_voice = load_data_test()
    # # 迭代打印 common_voice 前10个样本
    # for i in range(10):
    #     print(common_voice["train"][i])

    # common_voice_hi_process()
    # load_data_test()
    # common_voice = DatasetCommonVoiceHindi().load_processed_data()
    # print("test")

    # common_voice_cn_process()

    # dataset = DatasetCommonVoiceCN().load_processed_data()
    # print("dataset loaded")

    # raw_dataset_cn_process()

    # common_voice_cn_process()

    # dataset = DatasetFinAudioCN().load_raw_data()
    # print(dataset)
    # for i in range(10):
    #     print(dataset["test"][i])
        # print(dataset["test"][i]["audio"]["sampling_rate"])

    from datasets import load_dataset
    from datasets import DatasetDict

    # Load the raw dataset
    dataset = DatasetDict()
    dataset["test"] = load_dataset(path="./data/finaudio_zh_test/raw_dataset/test", split="test")

    # 迭代打印 common_voice 前10个样本
    print(dataset)
    for i in range(10):
        print(dataset["test"][i]["sentence"])
        print(dataset["test"][i]["audio"]["sampling_rate"])


