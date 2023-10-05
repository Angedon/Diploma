import numpy as np
import pandas as pd
import os
import func


dir = r"C:\Users\Gleb1\Desktop\Comments\test"
urls = [r"https://tabiturient.ru/vuzu/knitu/",
        r"https://tabiturient.ru/vuzu/sfedu/",
        r"https://tabiturient.ru/vuzu/dvfu/",
        r"https://tabiturient.ru/vuzu/spbguap/",
        r"https://tabiturient.ru/vuzu/isu/",
        r"https://tabiturient.ru/vuzu/hse-spb/",
        r"https://tabiturient.ru/vuzu/mgimo/",
        r"https://tabiturient.ru/vuzu/hse-nn/",
        r"https://tabiturient.ru/vuzu/miigaik/",
        r"https://tabiturient.ru/vuzu/mgppu/"]


#datas = pd.DataFrame({})
#for u in urls:
#    texts, all_text = func.foo(u)
#    texts = func.delete_tags(texts)
#    marks = func.form_marks(all_text, func.get_name(u))
#    print()
#    print(f"url = {u}, len(marks) = {len(marks)}, len(texts) = {len(texts)}")
#    func.write(texts, marks, func.get_name(u), dir)


datas = pd.DataFrame({})
k = []
for f in os.listdir(dir):
    if (f[-3:] != 'txt') & (f[0:3] != 'all'):
        data = pd.read_csv(dir + os.sep + f)
        k.append(len(data))
        data['Mark'].replace(['positive', 'negative', 'medium'], [1, -1, 0], inplace=True)
        if len(datas) != 0:
            datas = pd.concat([datas, data])
        else:
            datas = data

t = list(datas.Text)
m = list(datas.Mark.values)
#func.write_txt(t, m, 'test_t', dir)
func.write(t, m, 'all', dir)