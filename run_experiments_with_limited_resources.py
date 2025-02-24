import pandas as pd
from pathlib import Path
import argparse
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from utility_functions import *
import time


def temp(topkList):
    accuracy_dicionary = dict()
    for i in topkList:
        accuracy_dicionary["MRR@"+str(i)] = MRR(i)
        accuracy_dicionary["HR@"+str(i)] = HR(i)

    return accuracy_dicionary

def getResultDataDigi(targetItemList, topKList):

    main_dict = dict()
    gru4rec = pd.read_csv(str(Path("results/diginetica/gru4rec/test_single_gru4rec_diginetica-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GRU4REC"] =  calAccuracyForAllSessions(gru4rec, accuracy_dicionary)

    stamp = pd.read_csv(str(Path("results/diginetica/stamp/test_single_stamp_diginetica-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["STAMP"] =  calAccuracyForAllSessions(stamp, accuracy_dicionary)
    
    narm = pd.read_csv(str(Path("results/diginetica/narm/test_single_narm_diginetica-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["NARM"] =  calAccuracyForAllSessions(narm, accuracy_dicionary)
    
    srgnn = pd.read_csv(str(Path("results/diginetica/srgnn/test_single_srgnn_diginetica-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["SRGNN"] =  calAccuracyForAllSessions(srgnn, accuracy_dicionary)
    
    gcegnn = pd.read_csv(str(Path("results/diginetica/gcegnn/test_single_gcegnn_diginetica-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GCEGNN"] =  calAccuracyForAllSessions(gcegnn, accuracy_dicionary)
    
    tagnn = pd.read_csv(str(Path("results/diginetica/tagnn/test_single_tagnn_diginetica-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["TAGNN"] =  calAccuracyForAllSessions(tagnn, accuracy_dicionary)
    
    gnrrw = pd.read_csv(str(Path("results/diginetica/gnrrw/test_single_gnrrw_diginetica-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GNRRW"] =  calAccuracyForAllSessions(gnrrw, accuracy_dicionary)
    
    cotrec = pd.read_csv(str(Path("results/diginetica/cotrec/test_single_COTREC_diginetica-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["COTREC"] =  calAccuracyForAllSessions(cotrec, accuracy_dicionary)

    flcsp = pd.read_csv(str(Path("results/diginetica/flcsp/test_single_flcsp_diginetica-Saver@50.csv")), sep = ";")
    flcsp["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["FLCSP"] =  calAccuracyForAllSessions(flcsp, accuracy_dicionary)
    
    main_dict = pd.DataFrame.from_dict(main_dict, orient='index')
    main_dict.sort_values(by="MRR@10", ascending=False, inplace=True)
    print(main_dict)

    
def getResultDataRetail(targetItemList, topKList):
    main_dict = dict()
    gru4rec = pd.read_csv(str(Path("results/retailrocket/gru4rec/test_single_gru4rec_retailrocket-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GRU4REC"] =  calAccuracyForAllSessions(gru4rec, accuracy_dicionary)
    
    stamp = pd.read_csv(str(Path("results/retailrocket/stamp/test_single_stamp_retailrocket-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["STAMP"] =  calAccuracyForAllSessions(stamp, accuracy_dicionary)
    
    narm = pd.read_csv(str(Path("results/retailrocket/narm/test_single_narm_retailrocket-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["NARM"] =  calAccuracyForAllSessions(narm, accuracy_dicionary)
    
    srgnn = pd.read_csv(str(Path("results/retailrocket/srgnn/test_single_gnn_retailrocket-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["SRGNN"] =  calAccuracyForAllSessions(srgnn, accuracy_dicionary)
    
    gcegnn = pd.read_csv(str(Path("results/retailrocket/gcegnn/test_single_gcegnn_retailrocket-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GCEGNN"] =  calAccuracyForAllSessions(gcegnn, accuracy_dicionary)
    
    tagnn = pd.read_csv(str(Path("results/retailrocket/tagnn/test_single_tagnn_retailrocket-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["TAGNN"] =  calAccuracyForAllSessions(tagnn, accuracy_dicionary)
    
    gnrrw = pd.read_csv(str(Path("results/retailrocket/gnrww/test_single_gnrww_retailrocket-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GNRRW"] =  calAccuracyForAllSessions(gnrrw, accuracy_dicionary)
    
    cotrec = pd.read_csv(str(Path("results/retailrocket/cotrec/test_single_cotrec_retailrocket-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["COTREC"] =  calAccuracyForAllSessions(cotrec, accuracy_dicionary)
    
    flcsp = pd.read_csv(str(Path("results/retailrocket/flcsp/test_single_flcsp_retailrocket-Saver@50.csv")), sep = ";")
    flcsp["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["FLCSP"] =  calAccuracyForAllSessions(flcsp, accuracy_dicionary)

    main_dict = pd.DataFrame.from_dict(main_dict, orient='index')
    main_dict.sort_values(by="MRR@10", ascending=False, inplace=True)
    print(main_dict)
    
def getResultDataRsc15(targetItemList, topKList):
    main_dict = dict()
    gru4rec = pd.read_csv(str(Path("results/rsc15/gru4rec/test_single_gru4rec_rsc15-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GRU4REC"] =  calAccuracyForAllSessions(gru4rec, accuracy_dicionary)

    stamp = pd.read_csv(str(Path("results/rsc15/stamp/test_single_stamp_rec15-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["STAMP"] =  calAccuracyForAllSessions(stamp, accuracy_dicionary)
    
    narm = pd.read_csv(str(Path("results/rsc15/narm/test_single_narm_rsc15-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["NARM"] =  calAccuracyForAllSessions(narm, accuracy_dicionary)

    srgnn = pd.read_csv(str(Path("results/rsc15/srgnn/test_single_srgnn_rec15-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["SRGNN"] =  calAccuracyForAllSessions(srgnn, accuracy_dicionary)
    
    gcegnn = pd.read_csv(str(Path("results/rsc15/gcegnn/test_single_gcegnn_rec15-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GCEGNN"] =  calAccuracyForAllSessions(gcegnn, accuracy_dicionary)
    
    tagnn = pd.read_csv(str(Path("results/rsc15/tagnn/test_single_tagnn_rec15-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["TAGNN"] =  calAccuracyForAllSessions(tagnn, accuracy_dicionary)
    
    gnrrw = pd.read_csv(str(Path("results/rsc15/gnrrw/test_single_gnrrw_rec15-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["GNRRW"] =  calAccuracyForAllSessions(gnrrw, accuracy_dicionary)
    
    cotrec = pd.read_csv(str(Path("results/rsc15/cotrec/test_single_cotrec_rec15-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["COTREC"] =  calAccuracyForAllSessions(cotrec, accuracy_dicionary)
    
    flcsp = pd.read_csv(str(Path("results/rsc15/flcsp/test_single_flcsp_rec15-Saver@50.csv")), sep = ";")
    flcsp["target"] = targetItemList
    accuracy_dicionary = temp(topKList)
    main_dict["FLCSP"] =  calAccuracyForAllSessions(flcsp, accuracy_dicionary)

    main_dict = pd.DataFrame.from_dict(main_dict, orient='index')
    main_dict.sort_values(by="MRR@10", ascending=False, inplace=True)
    print(main_dict)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='retailrocket', help="diginetica / retailrocket / rsc15")
    parser.add_argument('--topKList', type=float, default=[10, 20], help='learning rate')
    args = parser.parse_args()
    start = time.time()
    print("Experiments are running for "+args.dataset)
    if args.dataset == "diginetica":
        targetItemList, _ = getData(dataset=args.dataset)
        getResultDataDigi(targetItemList, args.topKList)
    elif args.dataset == "retailrocket":
        targetItemList, _ = getData(dataset=args.dataset)
        getResultDataRetail(targetItemList, args.topKList)
    elif args.dataset == "rsc15":
        targetItemList, _ = getData(dataset=args.dataset)
        getResultDataRsc15(targetItemList, args.topKList)
    else:
        pass
    print("Time requires  "+str(time.time() - start)+"s")