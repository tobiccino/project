#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
# from model.svm_model import SVMModel
from model.naive_bayes_model import NaiveBayesModel


class TextClassificationPredict(object):
    def __init__(self):
        self.clf = self.train()

    def train(self):
        #  train data
        train_data = []
        train_data.append({"feature": u"xin chào", "target": "chaohoi"})
        train_data.append({"feature": u"chào admin", "target": "chaohoi"})
        train_data.append({"feature": u"Chào bot", "target": "chaohoi"})
        train_data.append({"feature": u"Có ai không nhỉ?", "target": "chaohoi"})
        train_data.append({"feature": u"Có thể cho tôi hỏi được không?", "target": "chaohoi"})
        ######
        train_data.append({"feature": u"Tôi cần việc làm", "target": "timviec"})
        train_data.append({"feature": u"Có việc nào cho tôi không", "target": "timviec"})
        train_data.append({"feature": u"Tôi là sinh viên cần tìm việc", "target": "timviec"})
        train_data.append({"feature": u"Tôi đang tìm công việc phù hợp", "target": "timviec"})
        train_data.append({"feature": u"Bạn có công việc nào không?", "target": "timviec"})
        train_data.append({"feature": u"Cho tôi xem những công việc hiện có", "target": "timviec"})
        train_data.append({"feature": u"Tôi là giáo viên cần tìm việc dạy thêm", "target": "timviec"})


        ######
        train_data.append({"feature": u"cám ơn và tạm biệt", "target": "tambiet"})
        train_data.append({"feature": u"tạm biệt nhé", "target": "tambiet"})
        train_data.append({"feature": u"Cám ơn rất nhiều", "target": "tambiet"})
        train_data.append({"feature": u"Hẹn gặp lại", "target": "tambiet"})
        df_train = pd.DataFrame(train_data)

        # init model naive bayes
        model = NaiveBayesModel()

        clf = model.clf.fit(df_train["feature"], df_train.target)
        return clf