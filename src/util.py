import pandas as pd
import cudf
import glob
import pickle
import os
import gc
import numpy as np
from typing import List
from collections import Counter, defaultdict
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def save_file(file, path):
    with open(path, mode='wb') as f:
        pickle.dump(file,f)
    return

###この高階関数が素晴らしい！次以降のコンペでも使うぞ
def get_file(func, file_name, type, **kwargs):
    if check_exist_file(file_name):
        print()
        print(f"{file_name},\n already exist, now loading...")
        if type == "dict" or "array":
            file = load_file(file_name)
        elif type == "df_pkl":
            file = pd.read_pickle(file_name)
        elif type == "pqt":
            file = pd.read_parquet(file_name)
        else:
            assert()
    else:
        print()
        print("no exist, running...")
        file = func(**kwargs)
        if type == "dict" or "array":
            save_file(file, file_name)
        elif type == "df_pkl":
            file.to_pickle(file_name)
        elif type == "pqt":
            file.to_parquet(file_name)
        else:
            assert()
    return file

def load_file(path):
    with open(path, mode='rb') as f:
        file = pickle.load(f)
    return file

def check_exist_file(file_path):
    flg = os.path.isfile(file_path)
    return flg

def dataset_load(dataset, debug, INPUT_DIR):
  if dataset=='task1':
    df = pd.read_pickle(INPUT_DIR +'task1_dataset.pkl')
    if debug:
      df = df.sample(frac=0.01).reset_index(drop=True)

  elif dataset=='task2':
    df = pd.read_pickle(INPUT_DIR + 'task2_dataset.pkl')
    if debug:
      df = df.sample(frac=0.01).reset_index(drop=True)

  return df

def extract_uid(df, start_uid, end_uid):
  # 評価用のuidのみに絞る
  df = df.loc[(df.uid>=start_uid)&(df.uid<=end_uid)]
  return df

def train_test_split_func(df, train_start_day, train_end_day, test_start_day, test_end_day):
  # 学習用データとテスト用のデータに分ける
  train = df.loc[(df.d>=train_start_day)&(df.d<=train_end_day)].reset_index(drop=True)
  test = df.loc[(df.d>=test_start_day)&(df.d<=test_end_day)].reset_index(drop=True)
  return train, test

def train_valid_test_split_func(df, train_start_day, train_end_day, valid_start_day, valid_end_day, test_start_day, test_end_day):
  # 学習用データとテスト用のデータに分ける
  train = df.loc[(df.d>=train_start_day)&(df.d<=train_end_day)].reset_index(drop=True)
  valid = df.loc[(df.d>=valid_start_day)&(df.d<=valid_end_day)].reset_index(drop=True)
  test = df.loc[(df.d>=test_start_day)&(df.d<=test_end_day)].reset_index(drop=True)
  return train, valid, test


def add_lag_mesh(df, n_days_ago):
    drop_cols = []
    for i in range(1, n_days_ago+1):
        df[f"xy_lag{i}_t"] = df.groupby(["uid","t"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_t")

        df[f"xy_lag{i}_wd_t"] = df.groupby(["uid","wd","t"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_wd_t")

        df[f"xy_lag{i}_cumcount_head"] = df.groupby(["uid","cumcount_head"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_cumcount_head")
    
        df[f"xy_lag{i}_cumcount_diff"] = df.groupby(["uid","cumcount_diff"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_cumcount_diff")
    
        df[f"xy_lag{i}_wd_cumcount_head"] = df.groupby(["uid","wd","cumcount_head"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_wd_cumcount_head")
        
        df[f"xy_lag{i}_wd_cumcount_tail"] = df.groupby(["uid","wd","cumcount_tail"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_wd_cumcount_tail")
        
        df[f"xy_lag{i}_wd_cumcount_diff"] = df.groupby(["uid","wd","cumcount_diff"])["xy"].shift(i)
        drop_cols.append(f"xy_lag{i}_wd_cumcount_diff")
    return df, drop_cols

# koyamasan copy
def add_cumcount(df):
  df["cumcount_head"] = df.groupby(["uid","d"]).cumcount()
  df["cumcount_tail"] = df.groupby(["uid","d"]).cumcount(ascending=False)
  df["cumcount_diff"] = df["cumcount_tail"] - df["cumcount_head"]
  return df

def add_group_xx(df, window_size):
  df['group_t'] = df.t // window_size
  df['group_head'] = df.cumcount_head // window_size
  df['group_tail'] = df.cumcount_tail // window_size
  df['group_diff'] = df.cumcount_diff // window_size
  return df

def sum_dicts(uids, dict1, dict2):
    merged_dict = defaultdict(lambda: defaultdict(Counter))
    
    for uid in uids:
        for t in range(48):  # t ranges from 0 to 47
            if uid in dict1 and t in dict1[uid]:
                merged_dict[uid][t].update(dict1[uid][t])
            if uid in dict2 and t in dict2[uid]:
                merged_dict[uid][t].update(dict2[uid][t])
    
    return merged_dict

def sum_dicts_t_group_t(uids, dict_t, dict_group_t, window_size):
    merged_dict = defaultdict(lambda: defaultdict(Counter))
    
    for uid in uids:
        for t in range(48):  # t ranges from 0 to 47
            if uid in dict_t and t in dict_t[uid]:
                merged_dict[uid][t].update(dict_t[uid][t])
            if uid in dict_group_t and t in dict_group_t[uid]:
                for i in range(window_size):
                   merged_dict[uid][t//window_size+i].update(dict_group_t[uid][t//window_size])
    
    return merged_dict

def map_dict_to_df_t(wgt_dict, df): 
    def get_most_common_xy(uid, t, wgt_dict):
        cnt = wgt_dict.get(uid, {}).get(t, {})
        if cnt != {}:
            return cnt.most_common(1)[0][0]
        else:
            return np.NAN
    srs = np.vectorize(get_most_common_xy, otypes=[float])(df["uid"], df["t"], wgt_dict)
    return srs

def map_dict_to_df_group_t(wgt_dict, df): 
    def get_most_common_xy(uid, t, wgt_dict):
        cnt = wgt_dict.get(uid, {}).get(t, {})
        if cnt != {}:
            return cnt.most_common(1)[0][0]
        else:
            return np.NAN
    srs = np.vectorize(get_most_common_xy, otypes=[float])(df["uid"], df["group_t"], wgt_dict)
    return srs


def get_uid2most_xy(df):
    mode_df = df.groupby(["uid"])["xy"].apply(lambda x: x.mode()).reset_index()[["uid","xy"]]
    uid2most_xy = {uid:xy for uid, xy in zip(mode_df["uid"], mode_df["xy"])}
    return uid2most_xy

def calc_wgt_func(train, func_list, co_start_day, co_end_day, test_end_day):
    merged_wgt = Counter()
    uids = list(train.uid.unique())
    for func in func_list:
        func_wgt = func(train, co_start_day, co_end_day, test_end_day)
        merged_wgt = sum_dicts(uids, merged_wgt, func_wgt)
    return merged_wgt

def classifier_culc(y_test, test_pred_class):

  print('')
  print('recall_score macro:', recall_score(y_test, test_pred_class, average='macro'))
  print('recall_score micro:', recall_score(y_test, test_pred_class, average='micro'))
  print('precision_score macro:', precision_score(y_test, test_pred_class, average='macro'))
  print('precision_score micro:', precision_score(y_test, test_pred_class, average='micro'))
  print('f1_score macro:', f1_score(y_test, test_pred_class, average='macro'))
  print('f1_score micro:', f1_score(y_test, test_pred_class, average='micro'))
  print('')
