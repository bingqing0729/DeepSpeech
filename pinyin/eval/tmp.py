from xpinyin import Pinyin
import pandas as pd 
import re
p = Pinyin()
r='[，？。！’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
a = pd.read_csv('input.csv')

for i in a.index:
    print(i)
    clean = re.sub(r,'',a.iloc[i,2])
    pinyin = p.get_pinyin(clean,"",tone_marks='numbers')
    if len(pinyin)>50:
        a.iloc[i,2] = ''
        a.iloc[i,1] = ''
        continue
    a.iloc[i,2] = clean
    a.iloc[i,1] = pinyin

a.to_csv('input_clean.csv',index=False,encoding='utf-8')
