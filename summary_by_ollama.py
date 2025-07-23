import os, re, json, time, tqdm, tool, jieba, whisper, jiwer, zhconv
from rouge_chinese import Rouge
from zhon.hanzi import punctuation
import cn2an
from ollama import chat
from ollama import ChatResponse
import numpy as np

summary_prompt_start = '假设你是个高级秘书，下文我分{}将提供一份详细的会议记录，每行的格式是“开始时间-结束时间 说话人：说话内容”，请你帮我总结以上这段会议的内容: 1. 输出一份精简的会议摘要，要求逻辑通顺，在200-300字左右。2.输出一份详细的会议摘要，要求逻辑通顺，在800-1200字左右。'

def main_summarize(data_json, model_name, spilt_num, summary_json):
    sub_summary_prompt_start = '假设你是个高级秘书，下文我将提供一份详细的会议记录，每行的格式是“开始时间-结束时间 说话人：说话内容”，请你帮我总结以上这段会议的内容, 输出一份详细的会议摘要，要求逻辑通顺'
    brief_summary_prompt = '假设你是个高级秘书，下文我将提供关于一场会议中连续的几个子会议摘要，请你帮我在这些子会议摘要的基础上，总结出整场会议的内容，输出一份精简的会议摘要，要求逻辑通顺，在200-300字左右。'
    detail_summary_prompt = '假设你是个高级秘书，下文我将提供关于一场会议中连续的几个子会议摘要，请你帮我在这些子会议摘要的基础上，总结出整场会议的内容，输出一份详细的会议摘要，要求逻辑通顺，在800-1200字左右。'
    data_dict = tool.json2dic(data_json)
    summary_dict = {}
    for key, value in tqdm.tqdm(data_dict.items()):
        value_lines= value.split('\n')
        all_lines_num = len(value_lines)
        chunk_size = (all_lines_num + spilt_num - 1) // spilt_num
        sub_summary = ''
        for i in range(spilt_num):
            start = i * chunk_size
            end = min(start + chunk_size, all_lines_num)
            sub_value = '\n'.join(value_lines[start:end])
            ChatResponse = chat(model=model_name, messages=[
                {'role': 'user', 'content': '{}\n{}'.format(sub_summary_prompt_start, sub_value)}])
            sub_summary += '第{}/{}段会议摘要：\n{}\n'.format(i+1, spilt_num, ChatResponse.message.content)
            
        BriefResponse = chat(model=model_name, messages=[
                {'role': 'user', 'content': '{}\n{}'.format(brief_summary_prompt, sub_summary)}])   
        brief_reply = BriefResponse.message.content.strip().replace('（200-300字）', '')
        
        DetailResponse = chat(model=model_name, messages=[
                {'role': 'user', 'content': '{}\n{}'.format(detail_summary_prompt, sub_summary)}])   
        detail_reply = DetailResponse.message.content.strip().replace('（800-1200字）', '')

        summary_dict[key] = {
            'brief': brief_reply,
            'detail': detail_reply
        }
        tool.json2dic(summary_json, summary_dict)
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

if __name__ == '__main__':
    model_name = 'deepseek-reasoner'
    # 'qwen-long', , 'moonshot-v1-auto', 'gemini-2.0-flash', 'llama3.2', 'llama3.1'
    summary_json='/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_CSOBx3_by_whisper-large-v3_by_{}.json'.format(model_name)
    # main_summarize(
    #     data_json='/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/recognition_CSOBx3+gss_by_avsr.json', 
    #     model_name=model_name, 
    #     spilt_num=4,
    #     summary_json=summary_json)
    # clear_up_summary_json('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_gss_whisper_large-v3_by_{}.json'.format(model_name))
    # all_summary_label = tool.find_all_target_files('/disk3/hangchen/data/misp-meeting/eval', 'meeting_summary.json')
    # merged_label = merge_summary_json(all_summary_label)
    # tool.json2dic('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/meeting_summary.json', merged_label)
    # compute_rouge(reference_json='/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/meeting_summary.json', hypothesis_json=summary_json)
    # 
    # /disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_gss_whisper_large-v3_by_qwen-long_clear_rouge.json
    mean_rouge('/disk5/hangchen/pandora/egs/misp-meeting/eval_far_gss/summary_CSOBx3_by_whisper-large-v3_by_qwen-long_rouge.json')
