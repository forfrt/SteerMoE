"""
 * author Ruitao Feng
 * created on 15-07-2025
 * github: https://github.com/forfrt
"""

"""
 * author Ruitao Feng
 * created on 24-03-2025
 * github: https://github.com/forfrt
"""

import io
import logging
import os
import random
import copy
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset
from pydub import AudioSegment
from pydub.generators import WhiteNoise

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

import setting
from utils import CosFileServer

cos = CosFileServer(setting.COS_BUCKET, setting.COS_SECRET_ID,
                    setting.COS_SECRET_KEY)


def put_cos(f_parquet, meta_file_name, meta_file_id, client):
    try:
        #     上传parquet对象
        response_parquet = client.put_object(
            Bucket='sj-ai-tmp-files-1307267266',
            Body=f_parquet.getvalue(),
            Key=
            f'{setting.COS_SECRET_KEY}/{meta_file_id}_{meta_file_name.rsplit(".", 1)[-2]}_{meta_file_name.rsplit(".", 1)[-1]}_train.parquet',
            EnableMD5=False)
    except Exception as e:
        raise Exception(f'{meta_file_name}~上传parquet失败：{str(e)}')


def generate_timestamps(chunks, actual_start_time):
    text = ''
    for chunk in chunks:
        text += f"<|{round(((chunk['START_TIME']-actual_start_time)/1000)/ 0.02) * 0.02:.2f}|>{chunk['text']}<|{round(((chunk['END_TIME']-actual_start_time)/1000)/ 0.02) * 0.02:.2f}|>"

    logging.debug(f"actual_start_time: {actual_start_time}")
    logging.debug(f"generate_timestamps: {text}")

    return text


def generate_no_timestamps(chunks, actual_start_time):
    text = ''
    for chunk in chunks:
        text += f"{chunk['text']}"

    logging.debug(f"actual_start_time: {actual_start_time}")
    logging.debug(f"generate_no_timestamps: {text}")

    return text


def get_empty_segmentation(duration, music_li, noise_li, frame_rate, channels,
                           sample_width):
    if random.random()<0.5:
        empty_seg = WhiteNoise().to_audio_segment(duration=duration)
        logging.info(f"added white noise, duration: {duration}, empty_seg: {empty_seg.duration_seconds}")
    else:
        if random.random()<0.5:
            music_file=random.choice(music_li)
            music = AudioSegment.from_file(music_file)
            while int(music.duration_seconds*1000)<duration:
                music = AudioSegment.from_file(random.choice(music_li))
            rand_start = random.randint(
                0,
                int(music.duration_seconds*1000)-duration)
            empty_seg = music[rand_start:rand_start+duration]
            logging.info(f"added music from {music_file}, duration: {duration}, empty_seg: {empty_seg.duration_seconds}")
        else:
            noise_file=random.choice(noise_li)
            noise = AudioSegment.from_file(noise_file)
            while int(noise.duration_seconds*1000)<duration:
                noise = AudioSegment.from_file(random.choice(noise_li))
            rand_start = random.randint(
                0,
                int(noise.duration_seconds*1000)-duration)
            empty_seg = noise[rand_start:rand_start+duration]
            logging.info(f"added noise from {noise_file}, duration: {duration}, empty_seg: {empty_seg.duration_seconds}")

    empty_seg = empty_seg.set_frame_rate(frame_rate)
    empty_seg = empty_seg.set_channels(channels)
    empty_seg = empty_seg.set_sample_width(sample_width)
    empty_seg = empty_seg-random.randrange(20, 70)

    return empty_seg


# 生成数据集
def handle_voice_ori(audio,
                     df,
                     audio_file_id,
                     meta_file_type,
                     tag,
                     music_wav_li,
                     noise_wav_li,
                     noise_rate=0.05,
                     meta_file_id='test'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

    meta_file_type = 'wav' if meta_file_type=='m4a' or meta_file_type=='M4A' else meta_file_type

    # add noise to increase the generality of the dataset
    length_range = [0, 5, 10, 15, 25, 30]
    tmp_df_dict_li = df.to_dict('records')

    noise_delay = 0
    id_delay = 0
    last_end_time = 0

    new_audio = AudioSegment.empty()
    new_audio = new_audio.set_frame_rate(audio.frame_rate)
    new_audio = new_audio.set_channels(audio.channels)
    new_audio = new_audio.set_sample_width(audio.sample_width)
    new_df_records = []

    for id, row in enumerate(tmp_df_dict_li):
        gap = audio[last_end_time:int(row['START_TIME'])]
        segment = audio[int(row['START_TIME']):int(row['END_TIME'])]
        last_end_time = int(row['END_TIME'])

        start_time = int(row['START_TIME'])+noise_delay
        end_time = int(row['END_TIME'])+noise_delay

        # if False:
        if random.random()<=noise_rate:
            rand_duration = random.choice(length_range[1:])
            duration = int(
                (random.random()*rand_duration/2+rand_duration/2)*1000)
            empty_seg = get_empty_segmentation(duration, music_wav_li,
                                               noise_wav_li, audio.frame_rate,
                                               audio.channels,
                                               audio.sample_width)

            # white_noise = WhiteNoise().to_audio_segment(duration=duration)
            # white_noise = white_noise.set_frame_rate(audio.frame_rate)
            # white_noise = white_noise.set_channels(audio.channels)
            # white_noise = white_noise.set_sample_width(audio.sample_width)
            # white_noise = white_noise-random.randrange(20, 70)
            # logging.info(f"duration: {duration}, white_noise: {white_noise.duration_seconds}")

            new_audio += (gap+empty_seg+segment)
            noise_row = {
                'START_TIME': start_time,
                'END_TIME': start_time+duration,
                # 'SENTENCE_MANUAL': 'noise',
                'SENTENCE_MANUAL': ' ',
                'ROLE_NUM': 0,
            }
            new_df_records.append(noise_row)

            noise_delay += duration
            id_delay += 1

            row['START_TIME'] = start_time+duration
            row['END_TIME'] = end_time+duration
        else:
            new_audio += gap+segment

            row['START_TIME'] = start_time
            row['END_TIME'] = end_time

        new_df_records.append(row)

    new_audio += audio[last_end_time:]
    new_df = pd.DataFrame(new_df_records)

    # uncomment to test
    # logging.info(
    #     f"before noised | audio_file_id: {audio_file_id}, audio: {audio.duration_seconds}, df: {len(df)}"
    # )
    # logging.info(
    #     # f"after noised | audio_file_id: {audio_file_id}, new_audio: {new_audio.duration_seconds}, df: {len(new_df)}"
    #     f"after noised | audio_file_id: {audio_file_id}, new_audio: {new_audio.duration_seconds}, df: {len(df)}, id_delay: {id_delay}, noise_delay: {noise_delay}"
    # )

    # noised_output_path = f"/root/autodl-nas/ruitao/data/raw/noised/{tag}/{audio_file_id}"
    # audio.export(f"{noised_output_path}_nonoised.wav", format='wav')
    # new_audio.export(f"{noised_output_path}.wav", format='wav')
    # new_df.to_csv(f"{noised_output_path}.csv", index=False)
    # exit()

    audio = new_audio
    df=new_df

    if True:
        # try:
        # length_range = [0,5,10,15,25]
        # current_threshold = random.choice(length_range)
        length_range = [0, 5, 10, 15, 25, 30]
        weights = [1, 1, 1, 2, 3, 15]  # 权重设置

        # 使用 random.choices 按照权重随机选择一个值
        current_threshold = random.choices(length_range, weights=weights,
                                           k=1)[0]
        tmp_df_info = df.to_dict('records')
        tmp_start_time = None
        tmp_end_time = None
        last_end_time = 0
        tmp_chunks = []
        full_paths = []
        full_sentences = []
        noise = False

        for row in tmp_df_info:

            row['START_TIME'] = int(row['START_TIME'])
            row['END_TIME'] = int(row['END_TIME'])
            current_gap = row['START_TIME']-last_end_time
            # 如果当前句子大于30s，保存之前的结果，直接舍弃这一句
            if row['END_TIME']-row['START_TIME']>30*1000:
                if tmp_end_time is None:

                    last_end_time = row['END_TIME']
                    continue
                else:
                    tmp_text = generate_no_timestamps(tmp_chunks, tmp_start_time)
                    segment = audio[tmp_start_time:tmp_end_time]
                    f = io.BytesIO()
                    segment.export(f, format=meta_file_type)
                    full_paths.append({
                        'path':
                            f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                        'bytes':
                            f.getvalue()
                    })
                    full_sentences.append(tmp_text)
                    tmp_start_time = None
                    tmp_end_time = None
                    tmp_chunks = []
                    current_threshold = random.choice(length_range)

                    last_end_time = row['END_TIME']
                    continue
            # 初始化开始时间
            if tmp_start_time is None:
                # 随机在开头添加gap
                if row['END_TIME']-last_end_time<=30*1000:
                    if random.random()>0.5:
                        tmp_start_time = last_end_time
                    else:
                        tmp_start_time = row['START_TIME']
                else:
                    tmp_start_time = row['START_TIME']

            # 如果加上这一句超过阈值，就保存
            if row['END_TIME']-tmp_start_time>=current_threshold*1000:
                if row['END_TIME']-tmp_start_time<=30*1000:
                    tmp_end_time = row['END_TIME']
                    tmp_chunks.append({
                        'START_TIME': row['START_TIME'],
                        'END_TIME': row['END_TIME'],
                        'text': row['SENTENCE_MANUAL']
                    })
                    tmp_text = generate_no_timestamps(tmp_chunks, tmp_start_time)
                    segment = audio[tmp_start_time:tmp_end_time]
                    f = io.BytesIO()
                    segment.export(f, format=meta_file_type)
                    full_paths.append({
                        'path':
                            f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                        'bytes':
                            f.getvalue()
                    })
                    full_sentences.append(tmp_text)
                    tmp_start_time = None
                    tmp_end_time = None
                    tmp_chunks = []
                    current_threshold = random.choice(length_range)
                else:
                    if tmp_end_time is not None:
                        tmp_text = generate_no_timestamps(tmp_chunks,
                                                       tmp_start_time)
                        segment = audio[tmp_start_time:tmp_end_time]
                        f = io.BytesIO()
                        segment.export(f, format=meta_file_type)
                        full_paths.append({
                            'path':
                                f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                            'bytes':
                                f.getvalue()
                        })
                        full_sentences.append(tmp_text)
                        current_threshold = random.choice(length_range)
                    tmp_start_time = row['START_TIME']
                    tmp_end_time = row['END_TIME']
                    tmp_chunks = [{
                        'START_TIME': row['START_TIME'],
                        'END_TIME': row['END_TIME'],
                        'text': row['SENTENCE_MANUAL']
                    }]

            else:
                tmp_end_time = row['END_TIME']
                tmp_chunks.append({
                    'START_TIME': row['START_TIME'],
                    'END_TIME': row['END_TIME'],
                    'text': row['SENTENCE_MANUAL']
                })

            last_end_time = row['END_TIME']

        logging.info(
            f"GENERATE||{audio_file_id} Processed: sentence: {len(full_sentences)}{full_sentences}, full_paths: {len(full_paths)}, tmp_chunks: {tmp_chunks}"
        )

        audio_dataset = Dataset.from_dict({
            "audio": full_paths,
            "sentence": full_sentences
        }).cast_column("audio", Audio())
        f_parquet = io.BytesIO()
        audio_dataset.to_parquet(f_parquet)
        #保存到本地
        # with open(f'../data/train/origin_parquet/{tag}/{meta_file_id}_{meta_file_name.rsplit(".", 1)[-2]}_{meta_file_name.rsplit(".", 1)[-1]}_train.parquet','wb') as f:
        with open(
                f'/root/autodl-nas/ruitao/data/train/origin_parquet/{tag}/{audio_file_id}_train.parquet',
                'wb') as f:
            f.write(f_parquet.getvalue())
        return True
    else:
        # except Exception as e:
        # raise Exception(f'{meta_file_name}~生成数据集错误信息：{str(e)}')
        # print(f'{meta_file_name}~生成数据集错误信息：{str(e)}')
        return False


# 生成数据集
def handle_voice(audio, df, audio_file_id, meta_file_type, tag):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

    meta_file_type = 'wav' if meta_file_type=='m4a' or meta_file_type=='M4A' else meta_file_type
    # length_range = [0,5,10,15,25]
    # current_threshold = random.choice(length_range)
    length_range = [0, 5, 10, 15, 25, 30]
    weights = [1, 1, 1, 2, 3, 15]  # 权重设置

    # 使用 random.choices 按照权重随机选择一个值
    current_threshold = random.choices(length_range, weights=weights, k=1)[0]
    tmp_df_info = df.to_dict('records')
    tmp_start_time = None
    tmp_end_time = None
    last_end_time = 0
    tmp_chunks = []
    full_paths = []
    full_sentences = []

    last_appended_id = -2

    if True:
        # try:
        for row in tmp_df_info:
            row['START_TIME'] = int(row['START_TIME'])
            row['END_TIME'] = int(row['END_TIME'])

            # current_id=row['id']
            current_gap = row['START_TIME']-last_end_time
            # 如果当前句子大于30s，保存之前的结果，直接舍弃这一句
            if row['END_TIME']-row['START_TIME']>30*1000:
                if tmp_end_time is None:

                    last_end_time = row['END_TIME']
                    continue
                else:
                    tmp_text = generate_no_timestamps(tmp_chunks, tmp_start_time)
                    segment = audio[tmp_start_time:tmp_end_time]
                    f = io.BytesIO()
                    segment.export(f, format=meta_file_type)
                    full_paths.append({
                        'path':
                            f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                        'bytes':
                            f.getvalue()
                    })
                    full_sentences.append(tmp_text)
                    tmp_start_time = None
                    tmp_end_time = None
                    tmp_chunks = []
                    current_threshold = random.choice(length_range)

                    last_end_time = row['END_TIME']

                    last_appended_id = -2
                    continue
            # 初始化开始时间
            if tmp_start_time is None:
                # 随机在开头添加gap
                if row['END_TIME']-last_end_time<=30*1000:
                    if random.random()>0.5:
                        tmp_start_time = last_end_time
                    else:
                        tmp_start_time = row['START_TIME']
                else:
                    tmp_start_time = row['START_TIME']

            # if True
            # if current_id==last_appended_id+1: #! set the statement to True to revert the last_appended_id change

            # 如果加上这一句超过阈值，就保存
                if row['END_TIME']-tmp_start_time>=current_threshold*1000:
                    if row['END_TIME']-tmp_start_time<=30*1000:
                        tmp_end_time = row['END_TIME']
                        tmp_chunks.append({
                            'START_TIME': row['START_TIME'],
                            'END_TIME': row['END_TIME'],
                            'text': row['SENTENCE_MANUAL']
                        })

                        tmp_text = generate_no_timestamps(tmp_chunks,
                                                       tmp_start_time)
                        segment = audio[tmp_start_time:tmp_end_time]
                        f = io.BytesIO()
                        segment.export(f, format=meta_file_type)
                        full_paths.append({
                            'path':
                                f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                            'bytes':
                                f.getvalue()
                        })
                        full_sentences.append(tmp_text)
                        tmp_start_time = None
                        tmp_end_time = None
                        tmp_chunks = []
                        current_threshold = random.choice(length_range)

                        last_appended_id = -2
                    else:
                        # Pack the last appended chunks
                        if tmp_end_time is not None:
                            tmp_text = generate_no_timestamps(
                                tmp_chunks, tmp_start_time)
                            segment = audio[tmp_start_time:tmp_end_time]
                            f = io.BytesIO()
                            segment.export(f, format=meta_file_type)
                            full_paths.append({
                                'path':
                                    f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                                'bytes':
                                    f.getvalue()
                            })
                            full_sentences.append(tmp_text)
                            current_threshold = random.choice(length_range)

                        tmp_start_time = row['START_TIME']
                        tmp_end_time = row['END_TIME']
                        tmp_chunks = [{
                            'START_TIME': row['START_TIME'],
                            'END_TIME': row['END_TIME'],
                            'text': row['SENTENCE_MANUAL']
                        }]

                        # last_appended_id=current_id

                else:
                    tmp_end_time = row['END_TIME']
                    tmp_chunks.append({
                        'START_TIME': row['START_TIME'],
                        'END_TIME': row['END_TIME'],
                        'text': row['SENTENCE_MANUAL']
                    })

                    # last_appended_id=current_id

            else:
                if tmp_end_time is not None:
                    tmp_text = generate_no_timestamps(tmp_chunks, tmp_start_time)
                    segment = audio[tmp_start_time:tmp_end_time]
                    f = io.BytesIO()
                    segment.export(f, format=meta_file_type)
                    full_paths.append({
                        'path':
                            f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                        'bytes':
                            f.getvalue()
                    })
                    full_sentences.append(tmp_text)
                    current_threshold = random.choice(length_range)

                tmp_start_time = row['START_TIME']
                tmp_end_time = row['END_TIME']
                tmp_chunks = [{
                    'START_TIME': row['START_TIME'],
                    'END_TIME': row['END_TIME'],
                    'text': row['SENTENCE_MANUAL']
                }]

                # last_appended_id=current_id

            last_end_time = row['END_TIME']
        """ The Audio feature accepts as input:
        - A str: Absolute path to the audio file (i.e. random access is allowed).
        - A dict with the keys:
            path: String with relative path of the audio file to the archive file.
            bytes: Bytes content of the audio file.
        - A dict with the keys:
            path: String with relative path of the audio file to the archive file.
            array: Array containing the audio sample
            sampling_rate: Integer corresponding to the sampling rate of the audio sample.
        """

        # generate timestamps for the last chunks which may not get processed
        if tmp_end_time is not None:
            tmp_text = generate_no_timestamps(tmp_chunks, tmp_start_time)
            segment = audio[tmp_start_time:tmp_end_time]
            f = io.BytesIO()
            segment.export(f, format=meta_file_type)
            full_paths.append({
                'path': f'{tmp_start_time}_{tmp_end_time}.{meta_file_type}',
                'bytes': f.getvalue()
            })
            full_sentences.append(tmp_text)

        logging.info(
            f"GENERATE||{audio_file_id} Processed: sentence: {len(full_sentences)}{full_sentences}, full_paths: {len(full_paths)}, tmp_chunks: {tmp_chunks}"
        )
        audio_dataset = Dataset.from_dict({
            "audio": full_paths,
            "sentence": full_sentences
        }).cast_column("audio", Audio())
        f_parquet = io.BytesIO()
        audio_dataset.to_parquet(f_parquet)
        #保存到本地
        with open(
                f'/root/autodl-nas/ruitao/data/train/origin_parquet/{tag}/{audio_file_id}_train.parquet',
                'wb') as f:
            f.write(f_parquet.getvalue())
        return True
    else:
        # except Exception as e:
        # print(f'{audio_file_id}~生成数据集错误信息：{str(e)}')
        return False


def load_slience_data(music_dir_li, noise_dir_li):
    music_wav_li = []
    for music_dir in music_dir_li:
        for root, dirs, files in os.walk(music_dir):
            for file in files:
                if file.endswith('.wav'):
                    music = AudioSegment.from_file(os.path.join(root, file))
                    if music.duration_seconds>5:
                        music_wav_li.append(os.path.join(root, file))

    logging.info(
        f"loaded music slience data: len(music_wav_li): {len(music_wav_li)}, music_wav_li[:5]: {music_wav_li[:5]}"
    )

    noise_wav_li = []
    for noise_dir in noise_dir_li:
        for root, dirs, files in os.walk(noise_dir):
            for file in files:
                if file.endswith('.wav'):
                    noise = AudioSegment.from_file(os.path.join(root, file))
                    if noise.duration_seconds>5:
                        noise_wav_li.append(os.path.join(root, file))

    logging.info(
        f"loaded noise slience data: len(noise_wav_li): {len(noise_wav_li)}, noise_wav_li[:5]: {noise_wav_li[:5]}"
    )

    return music_wav_li, noise_wav_li


import argparse
import json

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_tag', required=True, help='Batch ID')
    args = parser.parse_args()

    if args.batch_tag:
        print(f"batch_tag from arugments: {args.batch_tag}")
        batch_tag = args.batch_tag
        # tmp_file_path=f"/root/autodl-nas/ruitao/data/raw/audio2clips/{batch_tag}/"
        tmp_file_path = f"/root/autodl-nas/ruitao/data/raw/{batch_tag}/"
    else:
        batch_tag = f'{setting.SOURCE_TAG}_{setting.BATCH_TAG}'
        tmp_file_path = f"/root/autodl-nas/ruitao/data/raw/{batch_tag}/"
        print(f"batch_tag from setting.json: {batch_tag}")

    music_dir_li = [
        '/root/autodl-tmp/ruitao/whisper_test/data/musan/music/fma-western-art',
        '/root/autodl-tmp/ruitao/whisper_test/data/musan/music/hd-classical',
        '/root/autodl-tmp/ruitao/whisper_test/data/musan/music/jamendo',
        '/root/autodl-tmp/ruitao/whisper_test/data/musan/music/rfm',
    ]
    noise_dir_li = ['/root/autodl-tmp/ruitao/whisper_test/data/musan/noise/']
    music_wav_li, noise_wav_li = load_slience_data(music_dir_li, noise_dir_li)
    logging.info(f"loaded music slience data complete")

    logging.info(f"loading data from: {tmp_file_path}")

    ori_parquet_dir = f'/root/autodl-nas/ruitao/data/train/origin_parquet/{batch_tag}'
    processed_parquet_dir = f'/root/autodl-nas/ruitao/data/train/processed_parquet/{batch_tag}_processed_in_batch'
    processed_parquet_prefix = f'/root/autodl-nas/ruitao/data/train/processed_parquet/'

    logging.info(f"ori_parquet_dir: {ori_parquet_dir}")

    if not os.path.exists(ori_parquet_dir):
        os.makedirs(ori_parquet_dir)

    audio_files = []
    json_files = []
    for filename in os.listdir(tmp_file_path):
        if filename.endswith(".mp3"):
            logging.info(f"filename: {filename} {Path(filename).stem}")
            abs_audio_filename = os.path.join(tmp_file_path,
                                              f"{Path(filename).name}")
            abs_json_filename = os.path.join(tmp_file_path,
                                             f"{Path(filename).stem}.json")

            if Path(abs_json_filename).exists():
                audio_files.append(abs_audio_filename)
                json_files.append(abs_json_filename)

    logging.info(
        f"audio_files: {audio_files}, len(audio_files): {len(audio_files)}")
    logging.info(
        f"json_files: {json_files}, len(json_files): {len(json_files)}")

    from collections import defaultdict
    batch_dict = defaultdict(list)

    for id, audio_file in enumerate(audio_files[:]):
        file_id = Path(audio_file).stem
        logging.info(f"id: {id}, file_id: {file_id}, audio_file: {audio_file}")
        json_file = json_files[audio_files.index(audio_file)]
        # try:
        #     audio = AudioSegment.from_file(
        #         audio_file,
        #         format=Path(audio_file).suffix)
        # except:
        #     try:
        #         audio = AudioSegment.from_file(
        #         audio_file,
        #             format='mp4')
        #     except Exception as e:
        #         raise Exception(f'{audio_file}~audio对象生成：{str(e)}')

        # mate_sample_rate = audio.frame_rate
        # meta_duration = audio.duration_seconds*1000

        try:
            df_json = pd.read_json(json_file)
            df_clip = pd.DataFrame(list(df_json.result))
        except:
            #         遇到pandas读不了的json
            with open(json_file, 'rb') as f:
                data = json.load(f)
            df_clip = pd.DataFrame(list(data['result']))

        df_clip.rename(columns={
            'end_time': 'END_TIME',
            'role': 'ROLE_NUM',
            'sentence': 'SENTENCE_MANUAL',
            'start_time': 'START_TIME'
        },
                       inplace=True)

        # 语句标签nan替换为空字符串
        df_clip['SENTENCE_MANUAL'].fillna('', inplace=True)
        df_clip = df_clip[[
            'START_TIME', 'END_TIME', 'SENTENCE_MANUAL', 'ROLE_NUM'
        ]]
        logging.debug(f"df_clip: {df_clip}, len(df_clip): {len(df_clip)}")
        df_clip_dict = df_clip.to_dict('records')

        for row in df_clip_dict:
            batch_dict[file_id].append(row)

        # for k in batch_dict.keys():
        #     batch_dict[k].sort(key=lambda x:x['START_TIME'])

    logging.info(
        f"len(batch_dict): {len(batch_dict)}, batch_dict[list(batch_dict.keys())[0]]: {batch_dict[list(batch_dict.keys())[0]]}"
    )

    import tqdm
    error_list = []

    def handle_one_file(audio_file):
        audio_file_id = Path(audio_file).stem

        desti_file_path = f"{ori_parquet_dir}/{audio_file_id}.parquet"
        # desti_file_path = f'../data/train/origin_parquet/{batch_tag_dir}/{audio_file_id}_{audio_file_name.rsplit(".", 1)[-2]}_{audio_file_name.rsplit(".", 1)[-1]}_train.parquet'
        if os.path.exists(desti_file_path):
            return 0

        df_file = batch_dict[audio_file_id]
        if len(df_file)==0:
            return 0
        df_file = pd.DataFrame(df_file)

        logging.debug(
            f"audio_file_id: {audio_file_id}, df_file: {df_file.to_string()}, len(df_file): {len(df_file)}"
        )
        try:
            audio = AudioSegment.from_file(audio_file,
                                           format=audio_file.split('.')[-1])
            logging.info(
                f"ruitao audio_file: {audio_file}, audio: {type(audio)} {audio.duration_seconds, audio.frame_rate, audio.channels, audio.sample_width}"
            )
        except:
            logging.error(f'{audio_file_id}~音频文件读取失败')
            return 0

        handle_voice_ori(audio, df_file, audio_file_id,
                         audio_file.split('.')[-1], batch_tag, music_wav_li,
                         noise_wav_li, 0.05)

    import multiprocessing

    def process_list_in_parallel(ls):
        # 创建一个进程池
        with multiprocessing.Pool(processes=50) as pool:
            # 将列表的每个元素分配给进程池进行处理
            pool.map(handle_one_file, ls)

    process_list_in_parallel(audio_files[:])

    # 标记完成-
    with open(f"../status_folder/generate_parquet_{batch_tag}.done", "w") as f:
        f.write("done")
