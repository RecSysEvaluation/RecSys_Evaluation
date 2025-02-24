import pandas as pd
from pathlib import Path
import argparse
import numpy as np
import scipy.stats as stats
from utility_functions import *
import time

def getRecommendationDIGI(targetItemList):
    
    gru4rec = pd.read_csv(str(Path("results/diginetica/gru4rec/test_single_gru4rec_diginetica-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    
    narm = pd.read_csv(str(Path("results/diginetica/narm/test_single_narm_diginetica-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    
    gnrrw = pd.read_csv(str(Path("results/diginetica/gnrrw/test_single_gnrrw_diginetica-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList

    srgnn = pd.read_csv(str(Path("results/diginetica/srgnn/test_single_srgnn_diginetica-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList

    temp_dict = dict()
    temp_dict["gru4rec"] = gru4rec
    temp_dict["narm"] = narm
    temp_dict["gnrrw"] = gnrrw
    temp_dict["srgnn"] = srgnn

    accuracyForAlgorithm = dict()
    for i, value in temp_dict.items():
        accuracyForAlgorithm[i] = prepareDataforDifferentSessionLength(value, i)
    
    accuracyForAlgorithm = pd.DataFrame.from_dict(accuracyForAlgorithm, orient='index')
    print(accuracyForAlgorithm)
    accuracyForAlgorithm.to_csv(str(Path("results/diginetica/results_with_short_medium_long_sessions.txt")), sep = "\t")

def getRecommendationRSC15(targetItemList):
    gru4rec = pd.read_csv(str(Path("results/rsc15/gru4rec/test_single_gru4rec_rsc15-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    narm = pd.read_csv(str(Path("results/rsc15/narm/test_single_narm_rsc15-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    cotrec = pd.read_csv(str(Path("results/rsc15/cotrec/test_single_cotrec_rec15-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    gcegnn = pd.read_csv(str(Path("results/rsc15/gcegnn/test_single_gcegnn_rec15-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList

    
    temp_dict = dict()
    temp_dict["gru4rec"] = gru4rec
    temp_dict["narm"] = narm
    temp_dict["gnrrw"] = cotrec
    temp_dict["srgnn"] = gcegnn

    accuracyForAlgorithm = dict()
    for i, value in temp_dict.items():
        accuracyForAlgorithm[i] = prepareDataforDifferentSessionLength(value, i)
    accuracyForAlgorithm = pd.DataFrame.from_dict(accuracyForAlgorithm, orient='index')
    print(accuracyForAlgorithm)
    accuracyForAlgorithm.to_csv(str(Path("results/rsc15/results_with_short_medium_long_sessions.txt")), sep = "\t")


def getRecommendationRetail(targetItemList):
    gru4rec = pd.read_csv(str(Path("results/retailrocket/gru4rec/test_single_gru4rec_retailrocket-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    
    stamp = pd.read_csv(str(Path("results/retailrocket/stamp/test_single_stamp_retailrocket-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    
    srgnn = pd.read_csv(str(Path("results/retailrocket/srgnn/test_single_gnn_retailrocket-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList

    tagnn = pd.read_csv(str(Path("results/retailrocket/tagnn/test_single_tagnn_retailrocket-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList

    temp_dict = dict()
    temp_dict["gru4rec"] = gru4rec
    temp_dict["stamp"] = stamp
    temp_dict["srgnn"] = srgnn
    temp_dict["tagnn"] = tagnn

    accuracyForAlgorithm = dict()
    for i, value in temp_dict.items():
        accuracyForAlgorithm[i] = prepareDataforDifferentSessionLength(value, i)
    accuracyForAlgorithm = pd.DataFrame.from_dict(accuracyForAlgorithm, orient='index')
    print(accuracyForAlgorithm)
    accuracyForAlgorithm.to_csv(str(Path("results/retailrocket/results_with_short_medium_long_sessions.txt")), sep = "\t")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='retailrocket', help="diginetica / retailrocket / rsc15")
    args = parser.parse_args()

    start = time.time()
    print("Experiments are running for "+args.dataset)

    if args.dataset == "diginetica":
        targetItemList, data = getData(dataset=args.dataset)
        # Check number of short, medium and long sessions
        for i, value in short_med_longSessions(data).items():
            print(i +": "+str(len(value)))
        getRecommendationDIGI(targetItemList)
    
    elif args.dataset == "retailrocket":
        targetItemList, data = getData(dataset=args.dataset)
        # Check number of short, medium and long sessions
        for i, value in short_med_longSessions(data).items():
            print(i +": "+str(len(value)))
        targetItemList = getRecommendationRetail(targetItemList)
        
    elif args.dataset == "rsc15":
        targetItemList, data = getData(dataset=args.dataset)
        # Check number of short, medium and long sessions
        for i, value in short_med_longSessions(data).items():
            print(i +": "+str(len(value)))
        targetItemList = getRecommendationRSC15(targetItemList)
        
    else:
        pass

    print("Time requires  "+str(time.time() - start)+"s")