import pandas as pd
from pathlib import Path
import argparse
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from utility_functions import *
import time


def statisticalTest(gru4rec, stamp, narm, srgnn, gcegnn, tagnn, gnrrw, cotrec, flcsp, datasetName):
    f_statistic, p_value = stats.f_oneway(gru4rec, cotrec, flcsp, gcegnn, gnrrw, narm, srgnn, stamp, tagnn)
    # Show ANOVA results
    print("\nOne-Way ANOVA Results:")
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    # Interpretation
    if p_value < 0.05:
        print("\nConclusion: There is a significant difference between at least one of the models' accuracies.")
    else:
        print("\nConclusion: No significant difference found between the models' accuracies.")
    
    print("Conduct Post-hoc test: Tukey’s HSD")
    # Flatten the data and create corresponding group labels
    data = np.concatenate([gru4rec, stamp, narm, srgnn, gcegnn, tagnn, gnrrw, cotrec, flcsp])
    groups = (["GRU4Rec"] * len(gru4rec) +
             ["STAMP"] * len(stamp) +
             ["NARM"] * len(narm) + 
            ["SRGNN"] * len(srgnn) +
            ["GCEGNN"] * len(gcegnn) +
            ["TAGNN"] * len(tagnn) +
            ["GNRRW"] * len(gnrrw) +
            ["COTREC"] * len(cotrec) +
            ["FLCSP"] * len(flcsp)
            )
    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(data, groups, alpha=0.05)
    # Convert results to a DataFrame for better readability
    tukey_df = pd.DataFrame(data=tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])
    # Display Tukey's HSD results
    tukey_df.to_csv(str(Path("results/"+datasetName+"/results_tukey_test.txt")), sep = "\t")
    print("\nTukey's HSD Test Results:")
    print(tukey_df)
     

def getResultDataDigi(targetItemList, datasetName):
    gru4rec = pd.read_csv(str(Path("results/diginetica/gru4rec/test_single_gru4rec_diginetica-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    gru4rec = calAccuracyForEachSession(gru4rec, targetItemList)

    stamp = pd.read_csv(str(Path("results/diginetica/stamp/test_single_stamp_diginetica-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    stamp = calAccuracyForEachSession(stamp, targetItemList)

    narm = pd.read_csv(str(Path("results/diginetica/narm/test_single_narm_diginetica-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    narm = calAccuracyForEachSession(narm, targetItemList)
    
    srgnn = pd.read_csv(str(Path("results/diginetica/srgnn/test_single_srgnn_diginetica-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList
    srgnn = calAccuracyForEachSession(srgnn, targetItemList)

    gcegnn = pd.read_csv(str(Path("results/diginetica/gcegnn/test_single_gcegnn_diginetica-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList
    gcegnn = calAccuracyForEachSession(gcegnn, targetItemList)
    
    tagnn = pd.read_csv(str(Path("results/diginetica/tagnn/test_single_tagnn_diginetica-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList
    tagnn = calAccuracyForEachSession(tagnn, targetItemList)

    gnrrw = pd.read_csv(str(Path("results/diginetica/gnrrw/test_single_gnrrw_diginetica-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList
    gnrrw = calAccuracyForEachSession(gnrrw, targetItemList)
    
    cotrec = pd.read_csv(str(Path("results/diginetica/cotrec/test_single_COTREC_diginetica-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    cotrec = calAccuracyForEachSession(cotrec, targetItemList)

    flcsp = pd.read_csv(str(Path("results/diginetica/flcsp/test_single_flcsp_diginetica-Saver@50.csv")), sep = ";")
    flcsp["target"] = targetItemList
    flcsp = calAccuracyForEachSession(flcsp, targetItemList)
    statisticalTest(gru4rec, stamp, narm, srgnn, gcegnn, tagnn, gnrrw, cotrec, flcsp, datasetName)


def getResultDataRetail(targetItemList, datasetName):
    gru4rec = pd.read_csv(str(Path("results/retailrocket/gru4rec/test_single_gru4rec_retailrocket-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    gru4rec = calAccuracyForEachSession(gru4rec, targetItemList)
    
    stamp = pd.read_csv(str(Path("results/retailrocket/stamp/test_single_stamp_retailrocket-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    stamp = calAccuracyForEachSession(stamp, targetItemList)

    narm = pd.read_csv(str(Path("results/retailrocket/narm/test_single_narm_retailrocket-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    narm = calAccuracyForEachSession(narm, targetItemList)
    
    srgnn = pd.read_csv(str(Path("results/retailrocket/srgnn/test_single_gnn_retailrocket-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList
    srgnn = calAccuracyForEachSession(srgnn, targetItemList)

    gcegnn = pd.read_csv(str(Path("results/retailrocket/gcegnn/test_single_gcegnn_retailrocket-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList
    gcegnn = calAccuracyForEachSession(gcegnn, targetItemList)

    tagnn = pd.read_csv(str(Path("results/retailrocket/tagnn/test_single_tagnn_retailrocket-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList
    tagnn = calAccuracyForEachSession(tagnn, targetItemList)
    
    gnrrw = pd.read_csv(str(Path("results/retailrocket/gnrww/test_single_gnrww_retailrocket-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList
    gnrrw = calAccuracyForEachSession(gnrrw, targetItemList)

    cotrec = pd.read_csv(str(Path("results/retailrocket/flcsp/test_single_flcsp_retailrocket-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    cotrec = calAccuracyForEachSession(cotrec, targetItemList)

    flcsp = pd.read_csv(str(Path("results/retailrocket/flcsp/test_single_flcsp_retailrocket-Saver@50.csv")), sep = ";")
    flcsp["target"] = targetItemList
    flcsp = calAccuracyForEachSession(flcsp, targetItemList)
    statisticalTest(gru4rec, stamp, narm, srgnn, gcegnn, tagnn, gnrrw, cotrec, flcsp, datasetName)


def getResultDataRsc15(targetItemList, datasetName):
    gru4rec = pd.read_csv(str(Path("results/rsc15/gru4rec/test_single_gru4rec_rsc15-Saver@50.csv")), sep = ";")
    gru4rec["target"] = targetItemList
    gru4rec = calAccuracyForEachSession(gru4rec, targetItemList)

    stamp = pd.read_csv(str(Path("results/rsc15/stamp/test_single_stamp_rec15-Saver@50.csv")), sep = ";")
    stamp["target"] = targetItemList
    stamp = calAccuracyForEachSession(stamp, targetItemList)

    narm = pd.read_csv(str(Path("results/rsc15/narm/test_single_narm_rsc15-Saver@50.csv")), sep = ";")
    narm["target"] = targetItemList
    narm = calAccuracyForEachSession(narm, targetItemList)
    
    srgnn = pd.read_csv(str(Path("results/rsc15/srgnn/test_single_srgnn_rec15-Saver@50.csv")), sep = ";")
    srgnn["target"] = targetItemList
    srgnn = calAccuracyForEachSession(srgnn, targetItemList)

    gcegnn = pd.read_csv(str(Path("results/rsc15/gcegnn/test_single_gcegnn_rec15-Saver@50.csv")), sep = ";")
    gcegnn["target"] = targetItemList
    gcegnn = calAccuracyForEachSession(gcegnn, targetItemList)

    tagnn = pd.read_csv(str(Path("results/rsc15/tagnn/test_single_tagnn_rec15-Saver@50.csv")), sep = ";")
    tagnn["target"] = targetItemList
    tagnn = calAccuracyForEachSession(tagnn, targetItemList)

    gnrrw = pd.read_csv(str(Path("results/rsc15/gnrrw/test_single_gnrrw_rec15-Saver@50.csv")), sep = ";")
    gnrrw["target"] = targetItemList
    gnrrw = calAccuracyForEachSession(gnrrw, targetItemList)
    
    cotrec = pd.read_csv(str(Path("results/rsc15/cotrec/test_single_cotrec_rec15-Saver@50.csv")), sep = ";")
    cotrec["target"] = targetItemList
    cotrec = calAccuracyForEachSession(cotrec, targetItemList)
    
    flcsp = pd.read_csv(str(Path("results/rsc15/flcsp/test_single_flcsp_rec15-Saver@50.csv")), sep = ";")
    flcsp["target"] = targetItemList
    flcsp = calAccuracyForEachSession(flcsp, targetItemList)
    statisticalTest(gru4rec, stamp, narm, srgnn, gcegnn, tagnn, gnrrw, cotrec, flcsp, datasetName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='rsc15', help="diginetica / retailrocket / rsc15")
    args = parser.parse_args()
    start = time.time()
    print("Experiments are running for "+args.dataset)
    if args.dataset == "diginetica":
        targetItemList, _ = getData(dataset=args.dataset)
        getResultDataDigi(targetItemList, args.dataset)
    elif args.dataset == "retailrocket":
        targetItemList, _ = getData(dataset=args.dataset)
        getResultDataRetail(targetItemList, args.dataset)
    elif args.dataset == "rsc15":
        targetItemList, _ = getData(dataset=args.dataset)
        getResultDataRsc15(targetItemList, args.dataset)
    else:
        pass

    print("Time requires  "+str(time.time() - start)+"s")