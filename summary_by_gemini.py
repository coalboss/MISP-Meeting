import os, re, json, time, tqdm, tool, jieba, whisper, jiwer, zhconv
from openai import OpenAI
from rouge_chinese import Rouge
from zhon.hanzi import punctuation
import cn2an
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import pathlib
import textwrap

import google.generativeai as genai
import numpy as np

genai.configure(api_key="")

summary_prompt = '假设你是个高级秘书，下文我将提供一份详细的会议记录，每行的格式是“开始时间-结束时间 说话人：说话内容”，请你帮我总结以上这段会议的内容: 1. 输出一份精简的会议摘要，要求逻辑通顺，在200-300字左右。2.输出一份详细的会议摘要，要求逻辑通顺，在800-1200字左右。'

def main_summarize(data_json, model_name, summary_json):
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": summary_prompt},
        ]
    )
    data_dict = tool.json2dic(data_json)
    summary_dict = {}
    for key, value in tqdm.tqdm(data_dict.items()):
        response = chat.send_message(value)
        reply = response.text
        print(reply)
        # summary_list = re.findall('精简会议摘要([\s\S]*)详细会议摘要([\s\S]*)', reply)
        # brief_summary = summary_list[0][0].strip().replace('（200-300字）', '')
        # detailed_summary = summary_list[0][1].strip().replace('（800-1200字）', '')
        # summary_dict[key] = {
        #     'brief': brief_summary,
        #     'detail': detailed_summary
        # }
        # tool.json2dic(summary_json, summary_dict)
        # time.sleep(20)
        break
    return None

def clear_up_summary_json(summary_json):
    summary_dict = tool.json2dic(summary_json)
    for key, value in summary_dict.items():
        value = json.loads(value)
        all_summary = value['choices'][0]['message']['content'].replace('#', '')
        summary_list = re.findall('精简会议摘要([\s\S]*)详细会议摘要([\s\S]*)', all_summary)
        brief_summary = summary_list[0][0].strip().replace('（200-300字）', '')
        detailed_summary = summary_list[0][1].strip().replace('（800-1200字）', '')
        summary_dict[key] = {
            'brief': brief_summary,
            'detail': detailed_summary
        }
    updated_summary_json = summary_json.replace('.json', '_clear.json')
    tool.json2dic(updated_summary_json, summary_dict)
    return None

def merge_summary_json(summary_json_list):
    summary_dict = {}
    for summary_json in summary_json_list:
        session = os.path.split(os.path.split(summary_json)[0])[-1]
        summary_dict[session] = tool.json2dic(summary_json)
    return summary_dict

def word_segmentation(sentence):
    text_list = list(filter(lambda y: y!= '', re.split('|'.join(list(punctuation)+['\n']), sentence)))
    word_list = []
    for text_segmentation in text_list:
        word_list.extend(jieba.cut(text_segmentation, HMM=False))
    word_list = list(map(lambda y: cn2an.an2cn(y) if y.isdigit() else y, word_list))
    return word_list

def compute_rouge(reference_json, hypothesis_json):
    reference_dict = tool.json2dic(reference_json)
    hypothesis_dict = tool.json2dic(hypothesis_json)
    rouge = Rouge()
    rouge_dict = {}
    for key, value in tqdm.tqdm(reference_dict.items()):
        brief_reference = ' '.join(word_segmentation(value['brief']))
        detailed_reference = ' '.join(word_segmentation(value['detail']))
        detailed_hypothesis = ' '.join(word_segmentation(hypothesis_dict[key]['detail']))
        brief_hypothesis = ' '.join(word_segmentation(hypothesis_dict[key]['brief']))
        brief_scores = rouge.get_scores(brief_hypothesis, brief_reference)
        detailed_scores = rouge.get_scores(detailed_hypothesis, detailed_reference)
        rouge_dict[key] = {
            'brief': brief_scores,
            'detail': detailed_scores
        }
    tool.json2dic(hypothesis_json.replace('.json', '_rouge.json'), rouge_dict)
    return None

def compute_blue(reference_json, hypothesis_json):
    reference_dict = tool.json2dic(reference_json)
    hypothesis_dict = tool.json2dic(hypothesis_json)
    bleu_dict = {}
    for key, value in tqdm.tqdm(reference_dict.items()):
        brief_reference_list = [word_segmentation(value['brief'])]
        detailed_reference_list = [word_segmentation(value['detail'])]
        detailed_hypothesis_list = [word_segmentation(hypothesis_dict[key]['detail'])]
        brief_hypothesis_list = [word_segmentation(hypothesis_dict[key]['brief'])]
        brief_scores = corpus_bleu(brief_hypothesis_list, brief_reference_list)
        detailed_scores = corpus_bleu(detailed_hypothesis_list, detailed_reference_list, [1, 0, 0, 0])
        bleu_dict[key] = {
            'brief': brief_scores,
            'detail': detailed_scores
        }
        print(brief_scores)
        print(detailed_scores)
        break
    # tool.json2dic(hypothesis_json.replace('.json', '_blue.json'), bleu_dict)
    return None

def mean_rouge(rouge_json):
    brief_r_1_list = []
    brief_r_2_list = []
    brief_r_l_list = []
    detail_r_1_list = []
    detail_r_2_list = []
    detail_r_l_list = []
    rouge_dict = tool.json2dic(rouge_json)
    for key, value in rouge_dict.items():
        brief_r_1_list.append(value['brief'][0]['rouge-1']['f'])
        brief_r_2_list.append(value['brief'][0]['rouge-2']['f'])
        brief_r_l_list.append(value['brief'][0]['rouge-l']['f'])
        detail_r_1_list.append(value['detail'][0]['rouge-1']['f'])
        detail_r_2_list.append(value['detail'][0]['rouge-2']['f'])
        detail_r_l_list.append(value['detail'][0]['rouge-l']['f'])
    print(np.mean(brief_r_1_list)*100, np.mean(brief_r_2_list)*100, np.mean(brief_r_l_list)*100, np.mean(detail_r_1_list)*100, np.mean(detail_r_2_list)*100, np.mean(detail_r_l_list)*100)
    return None

if __name__ == '__main__':
    # model_name = 'gemini-1.5-flash'
    # main_summarize(
    #     data_json='/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/recognition_CSOBx3+gss_by_avsr.json', 
    #     model_name=model_name, 
    #     summary_json='/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_CSOBx3+gss_by_avsr_by_{}.json'.format(model_name))
    # clear_up_summary_json('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_gss_whisper_large-v3_by_{}.json'.format(model_name))
    # all_summary_label = tool.find_all_target_files('/disk3/hangchen/data/misp-meeting/eval', 'meeting_summary.json')
    # merged_label = merge_summary_json(all_summary_label)
    # tool.json2dic('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/meeting_summary.json', merged_label)
    summary_json = '/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_CSOBx3_by_whisper-large-v3_by_gemini-2.0-flash.json'
    compute_rouge(reference_json='/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/meeting_summary.json', hypothesis_json=summary_json)
    mean_rouge(rouge_json=summary_json.replace('.json', '_rouge.json'))
