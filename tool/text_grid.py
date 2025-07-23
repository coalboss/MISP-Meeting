#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import re, os, glob, jieba, codecs
import numpy as np
from zhon.hanzi import punctuation

def list_str_match(pattern_lst, str_lst):
    pattern_len = len(pattern_lst)
    if pattern_len != len(str_lst):
        raise ValueError('unmatched len of pattern lst {} and str lst {}'.format(pattern_len, len(str_lst)))
    value_lst =[]
    for i in range(pattern_len):
        value_candidate = re.findall(pattern_lst[i], str_lst[i])
        if len(value_candidate) == 1:
            value_lst.append(value_candidate[0])
        else:
            raise ValueError('unmatched pattern {} and str {}'.format(pattern_lst[i], str_lst[i]))
    return value_lst

def find_all_match_indices(string_list, pattern):
    indices = []
    for i, s in enumerate(string_list):
        if re.search(pattern, s):
            indices.append(i)
    return indices

class Interval(object):
    def __init__(self, idx=0, xmin=0., xmax=0., content=''):
        self.idx = idx
        self.xmin = xmin
        self.xmax = xmax
        self.content = content

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
        if self.idx < 0:
            raise ValueError('idx ({}) < 0'.format(self.idx))
    
    def numpy(self, sr=16000):
        sli_signs = ['<其他说话人>', '<NOISE>']
        point_num = int(round((self.xmax - self.xmin) * sr))
        if self.content in sli_signs:
            return np.zeros(point_num)
        else:
            return np.ones(point_num)
    
    def word_segmentation(self):
        no_word_signs = ['<其他说话人>', '<NOISE>', '<主说话人>', '<非会议内容>'] + ['*'*i for i in range(1, 30)]
        if self.content not in no_word_signs:
            text_list = list(filter(lambda y: y!= '', re.split('|'.join(list(punctuation)), self.content)))
            word_list = []
            for text_segmentation in text_list:
                word_list.extend(jieba.cut(text_segmentation, HMM=False))
            self.content = ' '.join(word_list)
        return None
    
    def text(self, prefix):
        return '{}-{:0>7d}-{:0>7d} {}'.format(
            prefix, int(1000*round(self.xmin, 3)), int(1000*round(self.xmax, 3)), self.content)
    
    def mask(self, mask_text='<Recognition Content>'):
        sli_signs = ['<其他说话人>', '<NOISE>', '<NOISE> ', '<OVERCLEAR>', '<knock>', '<DEAF>', 
                     '<moving  sound>', '<laughter>', '<sound of door>', '<其他说话人>', '<主说话人>', 
                     '<非会议内容>'] + ['*'*i for i in range(1, 30)]
        if self.content in sli_signs:
            return Interval(idx=self.idx, xmin=self.xmin, xmax=self.xmax, content=self.content)
        else:
            return Interval(idx=self.idx, xmin=self.xmin, xmax=self.xmax, content=mask_text)

class Tier(object):
    def __init__(self, idx=1, tclass='', name='', xmin=None, xmax=None, intervals=[]):
        self.idx = idx
        self.tclass = tclass
        self.name = name
        self.intervals = intervals
        self.xmin = xmin if xmin is not None else self.intervals[0].xmin
        self.xmax = xmax if xmax is not None else self.intervals[-1].xmax
        
        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
        
        x = self.xmin
        for i, interval in enumerate(self.intervals):
            if interval.xmin != x:
                if np.abs(interval.xmin - x) < 0.01:
                    self.intervals[i].max = x + interval.xmax - interval.xmin
                    self.intervals[i].min = x
                    x = interval.xmax
                else:
                    raise ValueError('NO.{} interval is not continuous, need {} but got {}'.format(i, x, interval.xmin))
            else:
                x = interval.xmax
        if x!= self.xmax:
            if np.abs(self.xmax - x) < 0.01:
                self.xmax = x
            else:
                raise ValueError('There is a gap between the last interval and the end of the Tier, from {} to{}'.format(x, self.xmax))
        if self.idx < 0:
            raise ValueError('idx ({}) < 0'.format(self.idx))

    def cutoff(self, xstart=None, xend=None, keep_idx=False):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        bias = xstart - self.xmin
        new_xmax = xend - bias
        new_xmin = self.xmin
        new_intervals = []
        new_idx = 1
        for interval in self.intervals:
            if interval.xmax <= xstart or interval.xmin >= xend:
                pass
            elif interval.xmin < xstart:
                if keep_idx:
                    used_idx = interval.idx
                else:
                    used_idx = new_idx   
                new_intervals.append(Interval(idx=used_idx, xmin=new_xmin, xmax=interval.xmax - bias, content=interval.content))
                new_idx += 1
            elif interval.xmax > xend:
                if keep_idx:
                    used_idx = interval.idx
                else:
                    used_idx = new_idx
                new_intervals.append(Interval(idx=used_idx, xmin=interval.xmin - bias, xmax=new_xmax, content=interval.content))
                new_idx += 1
            else:
                if keep_idx:
                    used_idx = interval.idx
                else:
                    used_idx = new_idx
                new_intervals.append(Interval(idx=used_idx, xmin=interval.xmin - bias, xmax=interval.xmax - bias, content=interval.content))
                new_idx += 1

        return Tier(idx=self.idx, tclass=self.tclass, name=self.name, xmin=new_xmin, xmax=new_xmax, intervals=new_intervals)
    
    def numpy(self, sr=16000):
        interval_arrays = []
        for interval in self.intervals:
            interval_arrays.append(interval.numpy(sr=sr))
        return np.concatenate(interval_arrays)
    
    def word_segmentation(self):
        for i in range(len(self.intervals)):
            self.intervals[i].word_segmentation()
        return None
    
    def text(self, prefix):
        no_word_signs = ['<其他说话人>', '<NOISE>', '<主说话人>', '<非会议内容>'] + ['*'*i for i in range(1, 30)]
        text_lines = []
        for interval in self.intervals:
            if interval.content not in no_word_signs:
                text_lines.append(interval.text(prefix=prefix))
        return text_lines
    
    def mask(self, mask_text='<Recognition Content>'):
        new_intervals = []
        for interval in self.intervals:
            new_intervals.append(interval.mask(mask_text=mask_text))
        return Tier(idx=self.idx, tclass=self.tclass, name=self.name, xmin=self.xmin, xmax=self.xmax, intervals=new_intervals)
        
# class definition
class TextGrid(object):
    def __init__(self, file_type='', object_class='', xmin=0., xmax=0., tiers=[]):
        self.file_type = file_type
        self.object_class = object_class
        self.tiers = tiers
        self.xmin = xmin if xmin is not None else self.tiers[0].xmin
        self.xmax = xmax if xmax is not None else self.tiers[0].xmax
    
        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
        for i, tier in enumerate(self.tiers[1:]):
            if tier.xmin != xmin or tier.xmax != xmax:
                raise ValueError('NO.{} tier is out of sync, should begin at {} but {} and end at {} but {}'.format(i, self.xmin, xmin, self.xmax, xmax))   
    
    def cutoff(self, xstart=None, xend=None, keep_idx=False):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        new_xmax = xend - xstart + self.xmin
        new_xmin = self.xmin
        new_tiers = []

        for tier in self.tiers:
            new_tiers.append(tier.cutoff(xstart=xstart, xend=xend, keep_idx=keep_idx))
        return TextGrid(file_type=self.file_type, object_class=self.object_class, xmin=new_xmin, xmax=new_xmax,
                        tiers=new_tiers)
    
    def numpy(self, sr=16000):
        for tier in self.tiers:
            if tier.name == '内容层':
                return tier.numpy(sr=sr)
    
    def word_segmentation(self):
        for i in range(len(self.tiers)):
            if self.tiers[i].name == '内容层':
                self.tiers[i].word_segmentation()
        return None
    
    def text(self, prefix, filepath=None, word_segmentation=False):
        used_i = 0
        for i in range(len(self.tiers)):
            if self.tiers[i].name == '内容层':
                used_i = i
                break
        
        used_tiers = self.tiers[used_i]
        if word_segmentation:
            used_tiers.word_segmentation()
        
        text_lines = used_tiers.text(prefix=prefix)
        if filepath is None:
            return text_lines
        else:
            store_dir = os.path.split(filepath)[0]
            if not os.path.exists(store_dir):
                os.makedirs(store_dir, exist_ok=True)
            processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), text_lines)]
            with codecs.open(filepath, 'w') as handle:
                handle.write(''.join(processed_lines))
            return None
    
    def mask(self, mask_text='<Recognition Content>'):
        new_tiers = []
        for tier in self.tiers:
            if tier.name == '内容层':
                new_tiers.append(tier.mask(mask_text=mask_text))
            else:
                new_tiers.append(tier)
        return TextGrid(file_type=self.file_type, object_class=self.object_class, xmin=self.xmin, xmax=self.xmax, tiers=new_tiers)
        
def read_textgrid_from_file(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        lines = list(filter(lambda y: y!= '', map(lambda x: x.strip().replace('"', ''), handle.readlines())))
    
    file_type, object_class, tg_xmin, tg_xmax, tiers_state, tiers_size = list_str_match(
            pattern_lst=[
                r'File\s*type\s*=\s*(\w+)', r'Object\s*class\s*=\s*(\w+)', r'xmin\s*=\s*(\d+\.?\d*)', 
                r'xmax\s*=\s*(\d+\.?\d*)', r'tiers\?\s*(.+)', r'size\s*=\s*(\d+\.?\d*)'], 
            str_lst=lines[:6])
    tg_xmin, tg_xmax, tiers_size = float(tg_xmin), float(tg_xmax), int(tiers_size)        
    
    tiers = []
    tiers_idxes = find_all_match_indices(lines, pattern=r'item\s*\[\d+\]:')
    assert len(tiers_idxes) == tiers_size
    tiers_idxes.append(len(lines))
    
    for i in range(tiers_size):
        tier_lines = lines[tiers_idxes[i]: tiers_idxes[i+1]]
        tier_idx, tclass, name, tier_xmin, tier_xmax, intervals_size =  list_str_match(
            pattern_lst=[r'item\s*\[(\d+)\]:', r'class\s*=\s*(\w+)', r'name\s*=\s*(\w+)', 
                         r'xmin\s*=\s*(\d+\.?\d*)', r'xmax\s*=\s*(\d+\.?\d*)', r'intervals:\s*size\s*=\s*(\d+\.?\d*)'], 
            str_lst=tier_lines[:6])
        tier_idx, tier_xmin, tier_xmax, intervals_size = int(tier_idx), float(tier_xmin), float(tier_xmax), int(intervals_size)
    
        intervals = []
        intervals_idxes = find_all_match_indices(tier_lines, pattern=r'intervals\s*\[\d+\]:')
        assert len(intervals_idxes) == intervals_size
        intervals_idxes.append(len(tier_lines))
        
        for j in range(intervals_size):
            intervals_idx, xmin, xmax, content =  list_str_match(
                pattern_lst=[r'intervals\s*\[(\d+)\]:', r'xmin\s*=\s*(\d+\.?\d*)', 
                             r'xmax\s*=\s*(\d+\.?\d*)', r'text\s*=\s*(.+)',], 
                str_lst=tier_lines[intervals_idxes[j]: intervals_idxes[j+1]])
            intervals_idx, xmin, xmax = int(intervals_idx), float(xmin), float(xmax)
            intervals.append(Interval(idx=intervals_idx, xmin=xmin, xmax=xmax, content=content))
        tiers.append(Tier(idx=tier_idx, tclass=tclass, name=name, xmin=tier_xmin, xmax=tier_xmax, intervals=intervals))
    tg = TextGrid(file_type=file_type, object_class=object_class, xmin=tg_xmin, xmax=tg_xmax, tiers=tiers)
    return tg

def write_textgrid_to_file(tg, filepath):
    floder = os.path.split(filepath)[0]
    os.makedirs(floder, exist_ok=True)
    with codecs.open(filepath, 'w', encoding='utf8') as handle:
        handle.write('File type = "{}"\n'.format(tg.file_type))
        handle.write('Object class = "{}"\n'.format(tg.object_class))
        handle.write('\n'.format(tg.object_class))
        
        tiers_size = len(tg.tiers)
        tiers_state = '<exists>' if tiers_size > 0 else '<not exists>'
        
        handle.write('xmin = {}\n'.format(tg.xmin))
        handle.write('xmax = {}\n'.format(tg.xmax))
        handle.write('tiers? {}\n'.format(tiers_state))
        handle.write('size = {}\n'.format(tiers_size))
        handle.write('item []:\n')
    
        for i, tier in enumerate(tg.tiers):
            handle.write('\titem [{}]:\n'.format(tier.idx))
            handle.write('\t\tclass = "{}"\n'.format(tier.tclass))
            handle.write('\t\tname = "{}"\n'.format(tier.name))
            handle.write('\t\txmin = {}\n'.format(tier.xmin))
            handle.write('\t\txmax = {}\n'.format(tier.xmax))
            handle.write('\t\tintervals: size = {}\n'.format(len(tier.intervals)))
            for j in range(len(tier.intervals)):
                handle.write('\t\t\tintervals [{}]:\n'.format(tier.intervals[j].idx))
                handle.write('\t\t\t\txmin = {}\n'.format(tier.intervals[j].xmin))
                handle.write('\t\t\t\txmax = {}\n'.format(tier.intervals[j].xmax))
                handle.write('\t\t\t\ttext = "{}"\n'.format(tier.intervals[j].content))
    return None


if __name__ == '__main__':
    checkout_tg = read_textgrid_from_file(filepath='F:\\MISP-Meeting\\eval\\M005\\M005-F8N\\M005-F8N-192.TextGrid')
    # print('xxxx')
    cut_tg = checkout_tg.cutoff(xstart=6.86, keep_idx=True)
    write_textgrid_to_file(filepath='F:\\MISP-Meeting\\eval\\M005\\M005-F8N\\M005-F8N-192_cut_6.68.TextGrid', tg=cut_tg)
