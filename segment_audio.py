#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, re, tool, codecs, tqdm
import numpy as np
from zhon.hanzi import punctuation
from scipy.io import wavfile


def file_segment(transcription_jsons, filetype):
    file_segment_dict = {}
    for transcription_json in tqdm.tqdm(transcription_jsons):
        file_root, file_name = os.path.split(transcription_json)
        file_root = os.path.split(file_root)[0]
        speaker = os.path.splitext(file_name)[0].split('-')[-1]
        transcription_dict = tool.json2dic(transcription_json)
        for key, value in tqdm.tqdm(transcription_dict.items(), leave=False):
            content = value['content']
            if bool(re.search(r'[\u4e00-\u9fa5]', content)):
                content = list(filter(lambda y: y!= '', re.split('|'.join(list(punctuation)), content)))
                content = ''.join(content)
                file_list = value[filetype]['file'].split('\\')
                session_name = file_list[0].split('-')[0]
                file_path = os.path.join(file_root, *file_list)
                duration = value[filetype]['duration']
                if file_path not in file_segment_dict:
                    file_segment_dict[file_path] = [{
                        'filename': '{}/{}-{}-Segments/{}-{}-{}-{}.wav'.format(file_root, session_name, filetype, speaker, session_name, filetype, key),
                        'text': content,
                        'start': duration[0],
                        'end': duration[1]
                    }]
                else:
                    file_segment_dict[file_path].append({
                        'filename': '{}/{}-{}-Segments/{}-{}-{}-{}.wav'.format(file_root, session_name, filetype, speaker, session_name, filetype, key),
                        'text': content,
                        'start': duration[0],
                        'end': duration[1]
                    })
    return file_segment_dict

# int(round(max((duration[0] - 0.2), 0)*fps)        

def segment_audio(file_segment_dict, fps):
    def read_pcm(pcm_file, channel=8, bit=32):
        with codecs.open(pcm_file, 'rb') as pcm_handle:
            pcm_frames = pcm_handle.read()
        np_array = np.frombuffer(pcm_frames, dtype='int{}'.format(bit), offset=0)
        np_array = np_array.reshape((-1, channel))
        return np_array
    
    result_dict = {}
    
    for key, value in tqdm.tqdm(file_segment_dict.items()):
        if key.split('.')[-1] == 'pcm':
            audio_data = read_pcm(key)[:, 0]
        elif key.split('.')[-1] == 'wav':
            _, audio_data = wavfile.read(key)
        else:
            raise ValueError('File type not supported')
        for item in tqdm.tqdm(value, leave=False):
            start = max(int(round((item['start'] - 0.2)*fps)), 0)
            end = min(int(round((item['end'] + 0.2)*fps)), len(audio_data))
            file_dir, file_name = os.path.split(item['filename'])
            file_name = os.path.splitext(file_name)[0]
            if not os.path.exists(item['filename']):
                segment_audio_data = audio_data[start:end]
                os.makedirs(file_dir, exist_ok=True)
                wavfile.write(item['filename'], 16000, segment_audio_data)

            result_dict[file_name] = {
                'wav': item['filename'],
                'text': item['text'], 
                'start': item['start'],
                'end': item['end']
            }
    
    return result_dict

if __name__ == '__main__':
    source_dir = '/disk3/hangchen/data/misp-meeting/eval/'
    transcription_jsons = tool.find_all_target_files(source_dir, '*-Transcription/*.json')
    file_type = 'F8N'
    file_segment_dict = file_segment(transcription_jsons, file_type)
    result_dict = segment_audio(file_segment_dict, fps=16000)
    tool.json2dic('/disk3/hangchen/data/misp-meeting/eval/index_{}_segments.json'.format(file_type), result_dict)
 
    