# -*- coding: utf-8 -*-
# @Time    : 2026/4/21 16:04
# @Author  : gaolei
# @FileName: test.py
# @Software: PyCharm
import os
import time
import config.codes as codes
import config.opportunity as op
import json
import re
# class ExpandedJSONEncoder(json.JSONEncoder):
#     def __init__(self, *args, **kwargs):
#         # 强制使用缩进
#         kwargs['indent'] = kwargs.get('indent', 4)
#         super().__init__(*args, **kwargs)
#
#     def encode(self, obj):
#         # 递归格式化函数
#         def format_obj(o, indent_level=0):
#             indent = ' ' * (indent_level * self.indent)
#             next_indent = ' ' * ((indent_level + 1) * self.indent)
#
#             if isinstance(o, dict):
#                 if not o:
#                     return '{}'
#                 items = []
#                 for k, v in o.items():
#                     key_repr = json.dumps(k, ensure_ascii=self.ensure_ascii)
#                     value_repr = format_obj(v, indent_level + 1)
#                     items.append(f'{next_indent}{key_repr}: {value_repr}')
#                 return '{\n' + ',\n'.join(items) + f'\n{indent}' + '}'
#             elif isinstance(o, list):
#                 if not o:
#                     return '[]'
#                 items = [format_obj(item, indent_level + 1) for item in o]
#                 return '[\n' + ',\n'.join(f'{next_indent}{item}' for item in items) + f'\n{indent}' + ']'
#             else:
#                 # 基本类型：字符串、数字、布尔、None
#                 return json.dumps(o, ensure_ascii=self.ensure_ascii)
#
#         return format_obj(obj, 0)
# # 使用自定义编码器

path=os.path.dirname(__file__)
path=os.path.join(path,'config','codes2.json')
with open(path, 'r', encoding='utf-8') as ff:
    data=json.load(ff)
pattern = r'("\d+":\s*\{\s*)\n\s*(.*?)\s*\n\s*(\},)'
replacement = r'\1\2\3'




for code,dt in codes.codes.items():
    if dt['industry']=='医药':
        data.get('创新药').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='航天':
        data.get('商业航天').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='电力':
        data.get('电网').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='算力租赁':
        data.get('算力').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='旅游':
        data.get('旅游').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='旅游':
        data.get('旅游').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='旅游':
        data.get('旅游').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})
    elif dt['industry']=='旅游':
        data.get('旅游').get('codes').setdefault(code,{'name':dt.get('name'),'5s':0,'hot':0,'pioneer':False})

with open(path, 'w+', encoding='utf-8') as f:
    json_str=json.dumps(data, ensure_ascii=False,indent=4)
    json_str =re.sub(pattern, replacement, json_str, flags=re.DOTALL)
    # json_str = re.sub(r'.*?"name', '"name', json_str)  # 处理对象起始
    json_str = re.sub(r'(?<!}),\s+', ', ', json_str)  # 处理逗号后空白（保留一个空格）
    # result = re.sub(r'(?<!\}),\s+', ', ', json_str)
    time.sleep(2)
    f.write(json_str)
#
    # data2=json.load(f)
    # time.sleep(2)