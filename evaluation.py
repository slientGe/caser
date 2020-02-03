import numpy as np
import math

def Metric_PrecN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(preds)

    return sum / count

def Metric_RecallN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(target)
    return sum / count

def cal_PR(target_list,predict_list,k=[1,5,10]):

    display_list = []

    for s in k:
        prec = Metric_PrecN(target_list,predict_list,s)
        recall = Metric_RecallN(target_list,predict_list,s)
        display = "Prec@{}:{:g} Recall@{}:{:g}".format(s,round(prec,4),s,round(recall,4))
        display_list.append(display)

    return ' '.join(display_list)
