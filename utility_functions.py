from pathlib import Path
import pandas as pd
import numpy as np
from statistics import mean 


def prepareDataforDifferentSessionLength(data, algoName):
    SessionsDfWithShort_Med_Long_dict = short_med_longSessions(data)
    temp_dict = dict()
    for i, value in SessionsDfWithShort_Med_Long_dict.items():
        df = data[data['SessionId'].isin(value)]
        if len(df) == 0:
            temp_dict[i] = 0
        else:
            temp_dict[i] = calAccuracyForAllSessions_(df)    
    return temp_dict

def getData(dataset = "rsc15"):
    if dataset == "diginetica":
          data_path = Path("data/"+dataset+"/fulltrain/diginetica_test.txt")
    if dataset == "retailrocket":
        data_path = Path("data/"+dataset+"/fulltrain/retailrocket_test.txt")
    if dataset == "rsc15":
        data_path = Path("data/"+dataset+"/fulltrain/rsc15By64/rsc15_test.txt")

    data = pd.read_csv(str(data_path), sep = "\t")
    data_group = data.groupby("SessionId")['ItemId'].agg(list)
    targetItem = list()
    for session_ in data_group:
        targetItem+=session_[1:]
    return targetItem, data

def calAccuracyForEachSession(df, targetItemList):
    recommendations = list(df["Recommendations"])
    hit = list()
    for i in range(len(targetItemList)):
        tempList = recommendations[i].split(",")[:20]
        tempList = [int(i) for i in tempList]
        if int(targetItemList[i]) in tempList:
             hit.append(1)
        else:
             hit.append(0)
    df_temp  = pd.DataFrame()
    df_temp["SessionId"] = df["SessionId"]
    df_temp["prediction"] = hit
    temp2 = df_temp.groupby("SessionId")['prediction'].agg(list)
    accuracyPerSession = list()
    for session_ in temp2:
        accuracyPerSession.append(sum(session_) / len(session_))     
    return accuracyPerSession

def calAccuracyForAllSessions_(df):
    recommendations = list(df["Recommendations"])
    targetItemList = list(df["target"])
    
    sum_ = 0
    hit = list()
    for i in range(len(targetItemList)):
        tempList = recommendations[i].split(",")[:20]
        tempList = [int(i) for i in tempList]
        if int(targetItemList[i]) in tempList:
             hit.append(1)
        else:
             hit.append(0)
        sum_ = sum_+1
    return sum(hit) / sum_ 


def calAccuracyForAllSessions(df, accuracyDict):
    recommendations = list(df["Recommendations"])
    score = list(df["Scores"])
    targetItemList = list(df["target"])
    
    
    for i in range(len(targetItemList)):
        temp_index = recommendations[i].split(",")
        temp_index = [int(i) for i in temp_index]
        
        temp_score = score[i].split(",")
        temp_score = [float(i) for i in temp_score]

        recomm_series = pd.Series(temp_score, index=temp_index) 

        for key in accuracyDict.keys():
            accuracyDict[key].add(recomm_series, targetItemList[i])
    
    tempScore = dict()
    for i in accuracyDict.keys():
            tempScore[i] = round(accuracyDict[i].score(), 3)

    return tempScore
def short_med_longSessions(data):
    data_group = data.groupby("SessionId")['SessionId'].size()
    SessionsDfWithShort_Med_Long_dict = dict()
    short_session = list()
    medium_sessions = list()
    long_sessions = list()
    for i, v in data_group.items():
        if( v <=3 ):
           short_session.append(i)
        elif(  v > 3 and  v <= 10):
            medium_sessions.append(i)
        else:
            long_sessions.append(i)
    
    SessionsDfWithShort_Med_Long_dict["short_sessions"] = short_session
    SessionsDfWithShort_Med_Long_dict["medium_sessions"] = medium_sessions
    SessionsDfWithShort_Med_Long_dict["long_sessions"] = long_sessions
    return SessionsDfWithShort_Med_Long_dict


class MRR: 
    
    def __init__(self, length=20):
        self.length = length
        self.MRR_score = []
    def add(self, recommendation_list, next_item):
        
        res = recommendation_list[:self.length]
        if next_item in res.index:
            rank = res.index.get_loc( next_item ) + 1
            self.MRR_score.append(1.0/rank)    
        else:
            self.MRR_score.append(0)
            
    def score(self):
        return mean(self.MRR_score)

    
class HR: 
    
    def __init__(self, length=20):
        self.length = length
        self.HR_score = []
        self.totat_sessionsIn_data = 0
        
    def add(self, recommendation_list, next_item):
        res = recommendation_list[:self.length]
        if next_item in res.index:
            self.HR_score.append(1.0)
        else:
            self.HR_score.append(0)
        
    def score(self):
        return mean(self.HR_score) 