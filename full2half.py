#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

"""
用来将全角字符转换成半角字符
"""

import re
def f2h_ch(ch):
    pattern = re.compile(u'[\uFF01-\uFF5E]+')
    if pattern.match(ch):
        return chr(ord(ch) - 0xFEE0)
    else:
        return ch

def f2h_text(text):
    pattern = re.compile(u'[\uFF01-\uFF5E]+')
    new_text = ''
    for w in text:
        if pattern.match(w):
            new_text += chr(ord(w) - 0xFEE0)
        else:
            new_text += w
    return new_text

