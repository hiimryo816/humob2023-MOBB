import pandas as pd
import cudf
import glob
import pickle
import os
import gc
import numpy as np
import pandas as pd, numpy as np
from tqdm import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
import cudf, itertools
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

def process_rows_t(chunk):
    partial_result = defaultdict(lambda: defaultdict(Counter))
    for _, row in chunk.iterrows():
        partial_result[row['uid']][row['t']][row['xy']] = row['wgt']
    return dict(partial_result)

def process_rows_group_t(chunk):
    partial_result = defaultdict(lambda: defaultdict(Counter))
    for _, row in chunk.iterrows():
        partial_result[row['uid']][row['group_t']][row['xy']] = row['wgt']
    return dict(partial_result)

def list_pqt_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pqt')]

def merge_dicts(dicts):
    merged = defaultdict(lambda: defaultdict(Counter))
    for d in dicts:
        for uid, t_dict in d.items():
            for t, counter in t_dict.items():
                for xy, wgt in counter.items():
                    merged[uid][t][xy] = wgt
    return dict(merged)

def td101(train, co_start_day, co_end_day, test_end_day):
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "group_t", "xy"]]
    uids = part_df.uid.unique()
    # weightの設定
    part_df["wgt"] = 1

    # weightの集計
    part_df = part_df.groupby(["uid","group_t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_group_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td002(train, co_start_day, co_end_day, test_end_day):
    # 平日を対象にIz, kを最適化した
    # 平日正解率は0.2927
    Iz = 1.81
    k =  0.0063
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td003(train, co_start_day, co_end_day, test_end_day):
    # 土日を対象にIz, kを最適化した
    # 土日正解率は0.2369
    Iz=1.0112
    k=0.0075
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td004(train, co_start_day, co_end_day, test_end_day):
    # 全体を対象にIz, kを最適化した
    # 正解率は0.2818
    Iz = 1.81
    k =  0.01182
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td105(train, co_start_day, co_end_day, test_end_day):
    # 全体を対象にIz, kを最適化した
    # t-> group_t
    # 正解率は0.2818
    Iz = 0.908
    k =  0.0178
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "group_t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","group_t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_group_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td006(train, co_start_day, co_end_day, test_end_day):
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    # weightの設定
    part_df["wgt"] = 1

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td007(train, co_start_day, co_end_day, test_end_day):
    # uid60000~69999
    # 平日を対象にIz=1固定でparam k, recall_macroを最適化した
    # timedecay_bo_t.ipynb
    Iz=1.0
    k=0.006614584754426877
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td008(train, co_start_day, co_end_day, test_end_day):
    # uid60000~69999
    # 平日を対象にIz=1固定でparam k, recall_macroを最適化した
    # timedecay_bo_t002.ipynb
    Iz=0.2
    k=0.018096837776122626
    part_df = train.loc[(train.d>30)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td009(train, co_start_day, co_end_day, test_end_day):
    # uid60000~69999
    # 平日を対象にIz=1固定でparam k, recall_macroを最適化した
    # timedecay_bo_t003.ipynb
    Iz=0.1
    k=0.01759355641432556
    part_df = train.loc[(train.d>30)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td010(train, co_start_day, co_end_day, test_end_day):
    # uid50000~59999
    # 平日を対象にIz=1固定でparam k, recall_macroを最適化した
    # timedecay_bo_t004.ipynb
    # 失敗作
    Iz=0.01
    k=0.011273965736562253
    part_df = train.loc[(train.d>10)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td011(train, co_start_day, co_end_day, test_end_day):
    # uid60000~69999
    # 平日を対象にIz=1固定でparam k, recall_macroを最適化した
    # timedecay_bo_t005.ipynb
    # 計算式を変えたよ
    Iz=0.5
    k=0.27065997170145306
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td012(train, co_start_day, co_end_day, test_end_day):
    # uid50000~59999
    # 平日を対象にIz=1固定でparam k, recall_microを最適化した
    # timedecay_bo_t006.ipynb
    # 計算式を変えたよ
    Iz=0.5
    k=0.3428520058104051
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td013(train, co_start_day, co_end_day, test_end_day):
    # uid40000~49999
    # 土日を対象にIz=1固定でparam k, recall_microを最適化した
    # timedecay_bo_t007.ipynb
    # 失敗作
    Iz=0.5
    k=0.11518946504962013
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td014(train, co_start_day, co_end_day, test_end_day):
    # uid40000~49999
    # 土日を対象にIz=1固定でparam k, recall_microを最適化した
    # timedecay_bo_t014.ipynb
    Iz=0.5
    k=0.19081248825025962
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt


####
def td108(train, co_start_day, co_end_day, test_end_day):
    # 全体を対象にIz, kを最適化した
    # だめだこりゃ
    # group_tのばあいは休日のみを対象にしないとダメかな
    # group_t
    Iz = 0.1
    k = 0.020912701020313573
    part_df = train.loc[(train.d>co_start_day)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "group_t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz*np.exp(-k * (test_end_day-part_df["d"]))

    # weightの集計
    part_df = part_df.groupby(["uid","group_t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_group_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td109(train, co_start_day, co_end_day, test_end_day):
    # uid40000~49999
    # 土日を対象にIz=1固定でparam k, recall_microを最適化した
    # timedecay_bo_t007.ipynb
    # 失敗作
    Iz=0.5
    k=0.11518946504962013
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "group_t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","group_t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_group_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td110(train, co_start_day, co_end_day, test_end_day):
    # uid40000~49999
    # 土日を対象にIz=1固定でparam k, recall_microを最適化した
    # timedecay_bo_t108.ipynb
    # 失敗作
    Iz=1
    k=0.10246380090167809
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "group_t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","group_t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_group_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt

def td111(train, co_start_day, co_end_day, test_end_day):
    # uid40000~49999
    # 全日を対象にIz=1固定でparam k, recall_microを最適化した
    # timedecay_bo_t108.ipynb
    Iz=1
    k=0.15056206232552458
    part_df = train.loc[(train.d>0)&(train.d<=co_end_day)]
    part_df = part_df[["uid", "d", "group_t", "xy"]]
    #
    # weightの設定
    part_df["wgt"] = Iz/(test_end_day-part_df["d"])**k

    # weightの集計
    part_df = part_df.groupby(["uid","group_t","xy"])["wgt"].sum().to_frame().reset_index()

    ### uid,t,xy:weightの辞書を作成
    # データフレームをチャンクに分割
    num_processes = 8
    chunks = [part_df.iloc[i::num_processes] for i in range(num_processes)]

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        partial_results = list(executor.map(process_rows_group_t, chunks))

    # 結果をマージ
    uid_t_xy2wgt = merge_dicts(partial_results)
    return uid_t_xy2wgt