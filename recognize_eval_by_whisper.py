#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, re, json, tqdm, tool, whisper, jiwer, zhconv

dir_map = {
    '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/misp_data/dev-part_wave_v6/far/wpe/gss_new/enhanced': '/disk3/hangchen/data/MISP2024-MMMT/eval-gss-audio/dev_part_enhanced',
    '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/misp_data/eval-part_wave_v6/far/wpe/gss_new/enhanced': '/disk3/hangchen/data/MISP2024-MMMT/eval-gss-audio/eval_part_enhanced'
}

def read_data_list(data_list):
    result = {}
    with open(data_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                line_value = json.loads(line)
                for key, value in dir_map.items():
                    line_value['wav'] = line_value['wav'].replace(key, value)
                result[line_value['key']] = {
                    'wav': line_value['wav'],
                    'text': line_value['txt']
                }
    return result

def main_recognize(data_dict, model_name, output_file):
    model = whisper.load_model(model_name, download_root='/disk5/hangchen/pandora/fork/whisper/downloads')
    result_dict = {}
    for key, value in tqdm.tqdm(data_dict.items()):
        result = model.transcribe(value['wav'])
        value['result'] = result['text']
        result_dict[key] = {
            **value, 'result': result['text']}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tool.json2dic(output_file, result_dict)
    return None

def compute_cer(result_file):
    result_dict = tool.json2dic(result_file)
    folder, file = os.path.split(result_file)
    filename = os.path.splitext(file)[0]
    cer_file = os.path.join(folder, filename + '_cer.json')
    eval_dict = {}
    all_subs, all_dels, all_ins, all_hits = 0, 0, 0, 0
    for key, value in tqdm.tqdm(result_dict.items()):
        reference = value.pop('text')
        hypothesis = zhconv.convert(value.pop('result'), 'zh-hans')
        output = jiwer.process_characters(reference, hypothesis)
        eval_line  = jiwer.visualize_alignment(output).split('\n')
        [(subs, dels, ins, hits)] = re.findall(r'substitutions=(\d+)\s*deletions=(\d+)\s*insertions=(\d+)\s*hits=(\d+)',eval_line[-4])
        subs, dels, ins, hits = int(subs), int(dels), int(ins), int(hits)
        all_subs += subs
        all_dels += dels
        all_ins += ins
        all_hits += hits
        [(cer)] = re.findall(r'cer=(\d+\.?\d*)%', eval_line[-2])
        cer = float(cer)
        eval_dict[key] = {
            **value,
            'reference': reference,
            'hypothesis': hypothesis,
            'substitutions': subs,
            'deletions': dels,
            'insertions': ins,
            'hits': hits,
            'cer': (subs + dels + ins)*100 / (subs + dels + hits)
        }
        if len(eval_line) == 5:
            eval_dict[key]['alignmnet'] = ' '*len(reference)
        else:
            eval_dict[key]['alignmnet'] = eval_line[3][5:]

    eval_dict['total'] = {
        'substitutions': all_subs,
        'deletions': all_dels,
        'insertions': all_ins,
        'hits': all_hits,
        'cer': (all_subs + all_dels + all_ins)*100 / (all_subs + all_dels + all_hits)
    }
    tool.json2dic(cer_file, eval_dict)   
    print('CER: {:.2f}%'.format(eval_dict['total']['cer'])) 
    return None

def reconstruct_meeting_from_text(text_file):
    eval_lines = tool.text2lines(text_file)
    session2meeting = {}
    for eval_line in eval_lines:
        key, value = eval_line.split(' ')
        speaker, session, _, _, start, end = key.split('_')
        speaker = speaker[1:]
        start = round(float(start)/100, 2)
        end = round(float(end)/100, 2)
        meeting_item = [
            start, 
            end, 
            speaker, 
            value
        ]
        if session not in session2meeting:
            session2meeting[session] = [meeting_item]
        else:
            session2meeting[session].append(meeting_item)
    
    for session, meeting in session2meeting.items():
        sorted_meeting = sorted(meeting, key=lambda x: x[0])
        meeting_content = ''
        for line in sorted_meeting:
            meeting_content += '{}-{} SPK{}: {}\n'.format(round(line[0], 2), round(line[1], 2), line[2], line[3])
        session2meeting[session] = meeting_content
    
    return session2meeting

def reconstruct_meeting_from_recognition(cer_file):
    eval_dict = tool.json2dic(cer_file)
    session2meeting = {}
    for key, value in eval_dict.items():
        if key == 'total':
            continue
        if len(key.split('_')) == 6:
            speaker, session, _, _, start, end = key.split('_')
            speaker = speaker[1:]
            start = round(float(start)/100, 2)
            end = round(float(end)/100, 2)
            meeting_item = [
                start, 
                end, 
                speaker, 
                value['']
            ]
        else:
            speaker, session, *_ = key.split('-')
            meeting_item = [
                value['start'], 
                value['end'], 
                speaker, 
                value['hypothesis']
            ]
        if session not in session2meeting:
            session2meeting[session] = [meeting_item]
        else:
            session2meeting[session].append(meeting_item)
    
    for session, meeting in session2meeting.items():
        sorted_meeting = sorted(meeting, key=lambda x: x[0])
        meeting_content = ''
        for line in sorted_meeting:
            meeting_content += '{}-{} SPK{}: {}\n'.format(round(line[0], 2), round(line[1], 2), line[2], line[3])
        session2meeting[session] = meeting_content
    
    return session2meeting

if __name__ == '__main__':
    # data_list = '/disk3/hangchen/data/MISP2024-MMMT/eval-gss-audio/eval_far_audio_segment/data.list'
    # data_dict = read_data_list(data_list)
    # for data_name in ['F8N']:
    #     for model_name in ['large-v3']:
    #         data_dict = tool.json2dic('/disk3/hangchen/data/misp-meeting/eval/index_{}_segments.json'.format(data_name))
            #  output_file = os.path.join('/disk3/hangchen/data/misp-meeting/eval/recognition_{}_by_whisper_{}'.format(data_name, model_name))
            #  meeting_dict = reconstruct_meeting_from_recognition(output_file+'_cer.json')
            #  tool.json2dic('oracle_meeting.json', meeting_dict)
            # output_dict = tool.json2dic(output_file)
            # new_output_dict = {}
            # for key, value in tqdm.tqdm(output_dict.items()):
            #     new_output_dict[key] = {
            #         **value,
            #         'start': data_dict[key]['start'],
            #         'end': data_dict[key]['end']
            #     }
            # tool.json2dic(output_file, new_output_dict)
            # main_recognize(data_dict, model_name, output_file)
            # compute_cer(output_file)
    # output_file = os.path.join('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/recognition_result_by_whisper_large-v3_cer.json')
    # meeting_dict = reconstruct_meeting_from_recognition(output_file)
    # tool.json2dic('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/recognition_result_by_whisper_large-v3_meeting.json', meeting_dict)
    # output_file = './eval_far_gss/recognition_CSOBx3_by_whisper-large-v3.json'
    # meeting_dict = reconstruct_meeting_from_recognition('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/recognition_result_by_whisper_large-v3_cer.json')
    meeting_dict = tool.json2dic('/disk3/hangchen/data/misp-meeting/eval/recognition_CSOBx3_by_whisper_large-v3_meeting.json')
    for key, value in meeting_dict.items():
        print(value, file=open('/disk3/hangchen/data/misp-meeting/eval/recognition_CSOBx3_by_whisper_large-v3_meeting_{}.txt'.format(key), 'w'))