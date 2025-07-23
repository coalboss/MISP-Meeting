#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import re, codecs
import numpy as np

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

class Interval(object):
    def __init__(self, xmin=0., xmax=0., text=''):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
    
    def numpy(self, sr=16000):
        sli_signs = ['<其他说话人>', '<NOISE>']
        point_num = int(round((self.xmax - self.xmin) * sr))
        if self.text in sli_signs:
            return np.zeros(point_num)
        else:
            return np.ones(point_num)

class Tier(object):
    def __init__(self, tclass='', name='', xmin=None, xmax=None, intervals=[]):
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
                raise ValueError('NO.{} interval is not continuous, need {} but got {}'.format(i, x, interval.xmin))
            else:
                x = interval.xmax
        if x!= self.xmax:
            raise ValueError('There is a gap between NO.{} interval and the end of the Tier, from {} to{}'.format(i, x, self.xmax))

    def cutoff(self, xstart=None, xend=None):
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
        for interval in self.intervals:
            if interval.xmax <= xstart or interval.xmin >= xend:
                pass
            elif interval.xmin < xstart:
                new_intervals.append(Interval(xmin=new_xmin, xmax=interval.xmax - bias, text=interval.text))
            elif interval.xmax > xend:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=new_xmax, text=interval.text))
            else:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=interval.xmax - bias, text=interval.text))

        return Tier(tier_class=self.tier_class, name=self.name, xmin=new_xmin, xmax=new_xmax, intervals=new_intervals)
    
    def numpy(self, sr=16000):
        interval_arrays = []
        for interval in self.intervals:
            interval_arrays.append(interval.numpy(sr=sr))
        return np.concatenate(interval_arrays)

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
    
    def cutoff(self, xstart=None, xend=None):
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
            new_tiers.append(tier.cutoff(xstart=xstart, xend=xend))
        return TextGrid(file_type=self.file_type, object_class=self.object_class, xmin=new_xmin, xmax=new_xmax,
                        tiers_status=self.tiers_status, tiers=new_tiers)
    
    def numpy(self, sr=16000):
        for tier in self.tiers:
            if tier.name == '内容层':
                return tier.numpy(sr=16000)
        
def read_textgrid_from_file(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        lines = list(filter(lambda y: y!= '', map(lambda x: x.strip().replace('"', ''), handle.readlines())))
    
    file_type, object_class, tg_xmin, tg_xmax, tiers_state, tiers_size = list_str_match(
            pattern_lst=[
                'File type = (\w+)', 'Object class = (\w+)', 'xmin = (\d+\.?\d*)', 
                'xmax = (\d+\.?\d*)', 'tiers\? (.+)', 'size = (\d+\.?\d*)'], 
            str_lst=lines[:6])
    tg_xmin, tg_xmax, tiers_size = float(tg_xmin), float(tg_xmax), int(tiers_size)        
    
    tiers = []
    tiers_idxes = []
    for i in range(tiers_size):
        tiers_idxes.append(lines.index('item [{}]:'.format(i + 1)))
    tiers_idxes.append(len(lines))
    
    for i in range(tiers_size):
        tier_lines = lines[tiers_idxes[i]+1: tiers_idxes[i+1]]
        tclass, name, tier_xmin, tier_xmax, intervals_size =  list_str_match(
            pattern_lst=[
                'class = (\w+)', 'name = (\w+)', 'xmin = (\d+\.?\d*)', 
                'xmax = (\d+\.?\d*)', 'intervals: size = (\d+\.?\d*)'], 
            str_lst=tier_lines[:5])
        tier_xmin, tier_xmax, intervals_size = float(tier_xmin), float(tier_xmax), int(intervals_size)
    
        intervals = []
        intervals_idxes = []
        for j in range(intervals_size):
            intervals_idxes.append(tier_lines.index('intervals [{}]:'.format(j + 1)))
        intervals_idxes.append(len(tier_lines))
        
        for j in range(intervals_size):
            xmin, xmax, text =  list_str_match(
                pattern_lst=[
                    'xmin = (\d+\.?\d*)', 'xmax = (\d+\.?\d*)', 'text = (.+)',], 
            str_lst=tier_lines[intervals_idxes[j]+1: intervals_idxes[j+1]])
            xmin, xmax = float(xmin), float(xmax)
            intervals.append(Interval(xmin=xmin, xmax=xmax, text=text))
        tiers.append(Tier(tclass=tclass, name=name, xmin=tier_xmin, xmax=tier_xmax, intervals=intervals))
    tg = TextGrid(file_type=file_type, object_class=object_class, xmin=tg_xmin, xmax=tg_xmax, tiers=tiers)
    return tg


if __name__ == '__main__':
    checkout_tg = read_textgrid_from_file(filepath='D:\\Code\\python_project\\Embedding_Aware_Speech_Enhancement_edition_3\\Textgrid_C0001\\1.TextGrid')
    cut_tg = checkout_tg.cutoff(xstart=220)
