#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from src.text_classification_predict import TextClassificationPredict

if __name__ == '__main__':
    tcp = TextClassificationPredict()

    while 1:
        text = raw_input("Nhập text: ")
        text = unicode(text, "utf-8")

        #  test data
        test_data = []
        test_data.append({"feature": text,})
        df_test = pd.DataFrame(test_data)

        result = pd.DataFrame(tcp.clf.predict_proba(df_test["feature"]), columns=tcp.clf.classes_)
        print result