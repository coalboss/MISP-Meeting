#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, glob, json, yaml, codecs, shutil
import numpy as np


def find_all_target_files(root_dir, target_file='*.insv'):
    sub_folder = []
    all_target_files = []
    while True:
        test_expression = os.path.join(root_dir, *sub_folder,'*')
        if len(glob.glob(test_expression)) > 0:
            find_expression = os.path.join(root_dir, *sub_folder, target_file)
            all_target_files.extend(glob.glob(find_expression))
            sub_folder.append('*')
        else:
            break
    return sorted(all_target_files)


def safe_copy(source, destination, keep_source=True, mode='cover'):
    """
    copy and paste file
    :param source: file be copy
    :param destination: file to paste
    :param keep_source: keep source or remove source
    :param mode: operation mode, ignore or cover
    :param other_params: reserved interface
    :return: None
    """
    def copy_process(origin, target, keep_origin):
        store_dir = os.path.split(target)[0]
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        if keep_origin:
            shutil.copy(origin, target)
        else:
            shutil.move(origin, target)
        return None

    if not os.path.exists(source):
        raise FileExistsError('no source file')
    else:
        if os.path.exists(destination):
            if mode == 'ignore':
                return False
            elif mode == 'cover':
                # os.remove(destination)
                copy_process(origin=source, target=destination, keep_origin=keep_source)
            else:
                raise NotImplementedError('unknown mode')
        else:
            copy_process(origin=source, target=destination, keep_origin=keep_source)
    return None


def pcm2numpy(pcm_file, bit_depth, offset):
    """
    read numpy array from pcm or sph file (sph: wav in TIMIT, offset=1024)
    :param sph_file: filepath of sph
    :param dtype: filepath of sph
    :return: numpy array of sph file
    """
    depth2dtype = {
        8: np.int16,
        16: np.int16,
        32: np.int32, 
        64: np.int64
    }
    assert bit_depth in depth2dtype, 'unknown bit_depth: {}'.format(bit_depth)
    
    with codecs.open(pcm_file, 'rb') as pcm_handle:
        pcm_bits = pcm_handle.read()
        
    frames_base = int(bit_depth / 8)
    frame_num, frames_remain = divmod(len(pcm_bits), frames_base)
    if frames_remain != 0:
        pcm_bits = pcm_bits[:-frames_remain]
        
    np_array = np.frombuffer(pcm_bits, dtype=depth2dtype[bit_depth], offset=offset)
    return np_array


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content)]
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None


def json2dic(jsonpath, dic=None, encoding='utf-8'):
    """
    read dic from json or write dic to json
    :param jsonpath: filepath of json
    :param dic: content dic or None, None means read
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(jsonpath, 'r', encoding=encoding) as handle:
            output = json.load(handle)
        return output
    else:
        assert isinstance(dic, dict)
        with codecs.open(jsonpath, 'w', encoding=encoding) as handle:
            json.dump(dic, handle, ensure_ascii=False, indent=4)
        return None

def yaml2dic(yamlpath, dic=None):
    """
    read dic from yaml or write dic to yaml
    :param yamlpath: filepath of yaml
    :param dic: content dic or None, None means read
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(yamlpath, 'r') as handle:
            return yaml.load(handle, Loader=yaml.FullLoader)
    else:
        with codecs.open(yamlpath, 'w') as handle:
            yaml.dump(dic, handle, indent=4)
        return None
    
if __name__ == '__main__':
    pass