import os
import polars as pl
import pandas as pd
from ptls.preprocessing import PandasDataPreprocessor
import numpy as np
from tqdm.notebook import tqdm
import gc
from datetime import datetime
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from ptls.data_load.utils import collate_feature_dict
from  sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
from sklearn.model_selection import KFold
from ptls.nn import TrxEncoder, RnnSeqEncoder, Head, LongformerSeqEncoder, TransformerSeqEncoder

def read_parquet(main_path):
    if main_path == 'empty_folder':
        return pl.DataFrame().lazy()
    list_pls = []
    for path in sorted(os.listdir(main_path)):
        list_pls += [ pl.scan_parquet(f'{main_path}/{path}') ]
    return pl.concat(list_pls)

def read_geo_parquet(main_path, dict_client_ids):
    if main_path == 'empty_folder':
        return pl.DataFrame().lazy()
    uniq_vals = list(dict_client_ids.keys())
    list_pls = []
    for path in tqdm(sorted(os.listdir(main_path))):
        tmp = pl.scan_parquet(f'{main_path}/{path}')
        tmp = tmp.select([pl.col(col_) for col_ in ['client_id', 'event_time', 'geohash_4']]).collect()
        tmp = tmp.filter(tmp['client_id'].is_in(uniq_vals))
        tmp = tmp.with_columns(tmp['client_id'].replace(dict_client_ids).cast(pl.datatypes.UInt32))
        tmp = tmp.with_columns((pl.col('event_time') + pl.duration(days=10)).dt.month_end().dt.date().cast( pl.datatypes.Int64))
        tmp = tmp.group_by(['client_id', 'event_time']).agg(pl.col('geohash_4').mode().map_elements(lambda x:x[0], return_dtype = pl.datatypes.UInt32))
        list_pls += [tmp]
    return pl.concat(list_pls)

def read_data(TRAIN_TARGET_PATH, TEST_TARGET_PATH, TRAIN_TXN_PATH, TEST_TXN_PATH, TRAIN_GEO_PATH, TEST_GEO_PATH, TRAIN_DIAL_PATH, TEST_DIAL_PATH, EMBED_LIST, TARGET_LIST ):
    target_train = read_parquet(TRAIN_TARGET_PATH).collect()
    target_test = read_parquet(TEST_TARGET_PATH).collect()

    target_test_add = pl.DataFrame()
    target_test_add = target_test_add.with_columns( pl.lit(target_test['client_id'].unique()).alias('client_id'))
    target_test_add = target_test_add.with_columns(pl.lit("2023-01-31").alias('mon'))
    target_test_add = target_test_add.with_columns([pl.lit(-1).alias(col) for col in TARGET_LIST])
    target_test_add = target_test_add[target_test.columns]

    target_data = pl.concat([target_train, target_test, target_test_add])

    transactions_train = read_parquet(TRAIN_TXN_PATH).collect()
    transactions_test = read_parquet(TEST_TXN_PATH).collect()

    uniq_vals = np.sort(target_data['client_id'].unique().to_numpy())
    dict_client_ids = {val:i for i,val in enumerate(uniq_vals)}
    if TRAIN_TXN_PATH != 'empty_folder':
        transactions_train = transactions_train.filter(pl.col('client_id').is_in(uniq_vals))
    transactions_test = transactions_test.filter(pl.col('client_id').is_in(uniq_vals))

    transactions_test = transactions_test.with_columns(transactions_test['client_id'].replace(dict_client_ids).cast(pl.datatypes.UInt32))
    if TRAIN_TXN_PATH != 'empty_folder':
        transactions_train = transactions_train.with_columns(transactions_train['client_id'].replace(dict_client_ids).cast(pl.datatypes.UInt32))

    target_data = target_data.with_columns(target_data['client_id'].replace(dict_client_ids).cast(pl.datatypes.UInt32))

    gc.collect()
    embed_max_data = {}
    for col in tqdm(EMBED_LIST):
        if TRAIN_TXN_PATH != 'empty_folder':
            max_val = max(transactions_train[col].max(), transactions_test[col].max())
        else:
            max_val = transactions_test[col].max()

        embed_max_data[col] = int(max_val)
        if max_val < 256:
            type_cast = pl.datatypes.UInt8
        elif max_val < 65535:
            type_cast = pl.datatypes.UInt16
        else:
            type_cast = pl.datatypes.UInt32
        transactions_test = transactions_test.with_columns(transactions_test[col].fill_null(0).cast(type_cast))
        if TRAIN_TXN_PATH != 'empty_folder':
            transactions_train = transactions_train.with_columns(transactions_train[col].fill_null(0).cast(type_cast))

    target_data = target_data.with_columns(pl.col("mon").str.to_datetime().dt.offset_by("-1mo").dt.month_end().dt.date().cast( pl.datatypes.Int64))

    transactions_test = transactions_test.with_columns((pl.col('event_time') + pl.duration(days=10)).dt.month_end().dt.date().cast( pl.datatypes.Int64).alias('mon') )
    if TRAIN_TXN_PATH != 'empty_folder':
        transactions_train = transactions_train.with_columns((pl.col('event_time') + pl.duration(days=10)).dt.month_end().dt.date().cast( pl.datatypes.Int64).alias('mon') )

    if TRAIN_TXN_PATH != 'empty_folder':
        transactions_train = transactions_train.with_columns((pl.col('event_time') + pl.duration(days=10)).dt.cast_time_unit("ns").cast( pl.datatypes.Int64) // 1000000000 )
    transactions_test = transactions_test.with_columns((pl.col('event_time') + pl.duration(days=10)).dt.cast_time_unit("ns").cast( pl.datatypes.Int64) // 1000000000)

    transactions = pl.concat([transactions_train, transactions_test])

    transactions = transactions.sort(['client_id', 'event_time'])
    target_data = target_data.sort(['client_id', 'mon'])

    dial_train = read_parquet(TRAIN_DIAL_PATH).collect()
    dial_test = read_parquet(TEST_DIAL_PATH).collect()

    if TRAIN_DIAL_PATH != 'empty_folder':
        dial_train = dial_train.filter(pl.col('client_id').is_in(uniq_vals))
    dial_test = dial_test.filter(pl.col('client_id').is_in(uniq_vals))

    dial_test = dial_test.with_columns(dial_test['client_id'].replace(dict_client_ids).cast(pl.datatypes.UInt32))
    if TRAIN_DIAL_PATH != 'empty_folder':
        dial_train = dial_train.with_columns(dial_train['client_id'].replace(dict_client_ids).cast(pl.datatypes.UInt32))

    if TRAIN_DIAL_PATH != 'empty_folder':
        dial_train = dial_train.with_columns((pl.col('event_time')).dt.cast_time_unit("ns").cast( pl.datatypes.Int64) // 1000000000 )
    dial_test = dial_test.with_columns((pl.col('event_time') ).dt.cast_time_unit("ns").cast( pl.datatypes.Int64) // 1000000000)

    dial = pl.concat([dial_train, dial_test])

    dial = dial.sort(['client_id', 'event_time'])

    del transactions_train, transactions_test, target_train, target_test, target_test_add, uniq_vals, dial_train, dial_test
    gc.collect()

    geo_train = read_geo_parquet(TRAIN_GEO_PATH,  dict_client_ids)
    geo_test = read_geo_parquet(TEST_GEO_PATH,  dict_client_ids)

    if TRAIN_DIAL_PATH != 'empty_folder':
        geo = pl.concat([geo_train, geo_test])
    else:
        geo = geo_test
    geo = geo.sort(['client_id', 'event_time'])
    del geo_train, geo_test,
    gc.collect()

    return transactions, target_data, embed_max_data, dict_client_ids, dial, geo

class SberDataset(Dataset):
    def __init__(self, data, data_key,  target, target_key, dialog, dialog_key, geo, geo_key, users, dict_cols, val = False, val_users = [],
                    embed_cols_train = [], TARGET_LIST = []):
        self.data_dict_cols, self.target_dict_cols, self.dialog_dict_cols, self.geo_dict_cols = dict_cols
        self.data = data
        self.data_key = data_key
        self.target = target
        self.target_key = target_key
        self.users = users
        self.val = val
        self.val_users = val_users
        self.empty_row = self.data[0].copy()
        self.dialog = dialog
        self.dialog_key = dialog_key
        self.geo = geo
        self.geo_key = geo_key
        self.TARGET_LIST = TARGET_LIST
        self.embed_cols_train  =embed_cols_train
        self.empty_row = self.data[0].copy()
        for i in range(1, len(self.empty_row)):
            self.empty_row[i] = np.array([0])
        prob_r = np.array([0.7, 0.5, 0.3])
        self.prob_0 = prob_r / prob_r.sum()
        self.prob_1 = prob_r[:-1] / prob_r[:-1].sum()
        self.choice = [1,2,3]
    def __getitem__(self, idx):
        client_id = self.users[idx]

        tmp_target = self.target[self.target_key[client_id]]

        EMPTY_FLAG = False
        if client_id in self.data_key:
            tmp_data =  self.data[self.data_key[client_id]]
        else:
            EMPTY_FLAG = True
            tmp_data = self.empty_row.copy()

        prob = self.prob_0
        choice = self.choice
        if client_id in self.val_users:
            choice = choice[1:]
            prob = self.prob_1

        month_target = int(np.random.choice(choice, size = 1, p = prob)[0])
        if self.val:
            month_target = 1


        data = {}
        for col in self.TARGET_LIST:
            idx_col = self.target_dict_cols[col]
            data[f'feat_{col}'] = torch.from_numpy(tmp_target[idx_col][:-month_target].copy()).float()
            data[col] = tmp_target[idx_col][-month_target]

        data['client_id'] = client_id

        pred_date = int(tmp_target[self.target_dict_cols['mon']][-month_target] * 24 * 60 * 60)
        len_ = len(tmp_data[self.data_dict_cols['event_time']])
        for i, x in enumerate(tmp_data[self.data_dict_cols['event_time']]):
            if x >= pred_date :
                len_ = i
                break
        if len_ == 0:
            EMPTY_FLAG = True
            tmp_data = self.empty_row.copy()
            len_ = 1

        WINDOW = 75
        LAST = 128
        FLAG_DROP = False

        FLAG_DROP = True

        len_embs = len(tmp_data[self.data_dict_cols[self.embed_cols_train[0]]][:len_])
        use = np.arange(len_embs)
        if len_embs > 2 and np.random.uniform() > 0.6 and not self.val:
            p = np.random.uniform(0, 0.2)
            numdr = np.random.randint(0, max(1, int(len_embs * p)))
            drop = np.random.choice(use, size = numdr, replace = False)
            use = np.setdiff1d(use, drop, assume_unique = True)

        for col in self.embed_cols_train + ['amount'] + ['event_time']:
            ind_col = self.data_dict_cols[col]
            if col == 'amount':
                data[col] = torch.from_numpy(tmp_data[ind_col][:len_][use][-LAST-WINDOW:]).float()
                # data['ma_' + col] = RollingOp( torch.mean, data[col], slice=WINDOW, axis=0 )[-LAST:]
                # data['counts'] = torch.from_numpy(self.count_len[:len_][-LAST:].copy()).float()
                data[col] = data[col][-LAST:]
                if not self.val and np.random.uniform() > 0.5:
                    data[col] = data[col] * np.random.uniform(0.9, 1.1, len(data[col]))
                    # data['ma_' + col] = data['ma_' + col] * np.random.uniform(0.9, 1.1, len(data['ma_' + col]))
            elif col == 'event_time':
                data[col] = torch.from_numpy(tmp_data[ind_col][:len_][use][-LAST:]).long()
                if EMPTY_FLAG:
                    data['diff_' + col]  = torch.from_numpy(np.array([0.5]) ).float()
                else:
                    data['diff_' + col] = (pred_date - tmp_data[ind_col][:len_].copy() )  / (24 * 60 * 60 * 360)
                    data['diff_' + col]  = data['diff_' + col][-LAST:] - 0.5
                    if not self.val and np.random.uniform() > 0.5:
                        data['diff_' + col] = data['diff_' + col] * np.random.uniform(0.9, 1.1, len(data['diff_' + col]))
                    data['diff_' + col]  = torch.from_numpy(data['diff_' + col] ).float()

            else:
                data[col] = torch.from_numpy(tmp_data[ind_col][:len_][use][-LAST:]).long()

        tmp_dial_time = np.array([0.5])
        tmp_dial_embs = np.zeros((1, 768))
        if client_id in self.dialog_key:
            tmp_dial = self.dialog[self.dialog_key[client_id]]
            len_ = len(tmp_dial[self.dialog_dict_cols['event_time']])
            for i, x in enumerate(tmp_dial[self.dialog_dict_cols['event_time']]):
                if x >= pred_date :
                    len_ = i
                    break

            if len_ != 0:
                tmp_dial_time = self.dialog[self.dialog_key[client_id]][self.dialog_dict_cols['event_time']][:len_].copy()
                tmp_dial_time = ((pred_date - tmp_dial_time )  / (24 * 60 * 60 * 360)) - 0.5
                # tmp_dial_time = tmp_dial_time[-20:]
                tmp_dial_time = tmp_dial_time[-1:]
                tmp_dial_embs = np.vstack(self.dialog[self.dialog_key[client_id]][self.dialog_dict_cols['embedding']][:len_])[-20:].copy()

        data['dialog'] = torch.from_numpy((np.concatenate([tmp_dial_embs.mean(0), tmp_dial_time]))).float()


        tmp_geo_hash = np.array([0.5])
        tmp_geo_time = np.array([0])
        if client_id in self.geo_key:
            tmp_geo = self.geo[self.geo_key[client_id]]
            len_ = len(tmp_geo[self.geo_dict_cols['event_time']])
            for i, x in enumerate(tmp_geo[self.geo_dict_cols['event_time']]):
                if x * (24 * 60 * 60) > pred_date :
                    len_ = i
                    break

            if len_ != 0:
                tmp_geo_time = self.geo[self.geo_key[client_id]][self.geo_dict_cols['event_time']][:len_].copy()
                tmp_geo_time = ((pred_date - tmp_geo_time * (24 * 60 * 60) )  / (24 * 60 * 60 * 360)) - 0.5
                # tmp_dial_time = tmp_dial_time[-20:]
                tmp_geo_hash =  self.geo[self.geo_key[client_id]][self.geo_dict_cols['geohash_4']][:len_].copy()


        data['geo_time'] = torch.from_numpy(tmp_geo_time).float()
        data['geo_hash'] = torch.from_numpy(tmp_geo_hash).long()

        return data

    def __len__(self):
        return len(self.users)


class Collate:
    def __init__(self, TARGET_LIST, embed_cols_train):
        self.embed_cols_train = embed_cols_train
        self.TARGET_LIST = TARGET_LIST
        tmp_dial_time = np.array([0.5])
        tmp_dial_embs = np.zeros((1, 768))
        self.empty_dial = torch.from_numpy(np.concatenate([tmp_dial_embs, tmp_dial_time[None].T], 1)).float()
        pass

    def __call__(self, batch):
        # max_len = max([len(sample['dialog']) for sample in batch])
        # padded_batch_pivot = [torch.cat([sample['dialog'], self.empty_dial.repeat( max_len - len(sample['dialog']), 1)]) for sample in batch]
        # padded_batch_pivot = torch.stack(padded_batch_pivot, 0).float()

        padded_batch_feat = collate_feature_dict([{k:v for k,v in sample.items() if k in self.embed_cols_train + [ 'amount', 'diff_event_time'] } for sample in batch])
        padded_batch_target = collate_feature_dict([{k:v for k,v in sample.items() if k.startswith('feat_target_') } for sample in batch])
        padded_batch_dialog = torch.vstack([sample['dialog'] for sample in batch]).float()
        padded_batch_geo = collate_feature_dict([{k:v for k,v in sample.items() if k in ['geo_time', 'geo_hash'] } for sample in batch])


        target = []
        for target_col_name in self.TARGET_LIST:
            target += [ torch.from_numpy(np.array([sample[target_col_name] for sample in batch])).float() ]

        return padded_batch_feat, padded_batch_target, padded_batch_dialog, padded_batch_geo, torch.vstack(target).T

def standart_split(data):
    split_list = []
    kf = KFold(n_splits = 5, random_state = 228, shuffle = True)
    for train_index, val_index in kf.split(data, data) :
        train_index = data[train_index]
        val_index = data[val_index]
        split_list += [(train_index, val_index)]
    return split_list


def val_step(sber_dataloader_val, model, device):
    preds = []
    targets = []
    loss_fct = nn.BCEWithLogitsLoss()
    model.eval()

    tk0 = tqdm(enumerate(sber_dataloader_val), total = len(sber_dataloader_val))
    average_loss = 0
    with torch.no_grad():
        for batch_number, (vals) in tk0:
            emb_feats = vals[0].to(device)
            emb_target = vals[1].to(device)
            emb_dialog = vals[2].to(device)
            emb_geo = vals[3].to(device)
            target = vals[4].to(device)
            with torch.cuda.amp.autocast():
                ans = model(emb_feats, emb_target, emb_dialog, emb_geo )
                loss = loss_fct(ans, target)
            average_loss += loss.cpu().detach().numpy()
            preds += [torch.sigmoid(ans ).cpu().detach().numpy()]
            targets += [target.cpu().detach().numpy()]
            tk0.set_postfix(loss=average_loss / (batch_number + 1), stage="val")
    sc = 0
    for k in range(4):
        pr = np.concatenate(preds)[:, k]
        tr = np.concatenate(targets)[:, k]
        sc += roc_auc_score(tr, pr)
        print(roc_auc_score(tr, pr))
    print('FULL ', sc / 4)



class SberModel(nn.Module):
    def __init__(self, trx_encoder_params, tgt_encoder_params, geo_encoder_params, num_labels):


        seq_encoder_feat = RnnSeqEncoder(
            trx_encoder=TrxEncoder(**trx_encoder_params),
            hidden_size=512,
            type='gru',
        )

        seq_encoder_tgt = RnnSeqEncoder(
            trx_encoder=TrxEncoder(**tgt_encoder_params),
            hidden_size=128,
            type='gru',
        )

        seq_encoder_geo = RnnSeqEncoder(
            trx_encoder=TrxEncoder(**geo_encoder_params),
            hidden_size=64,
            type='gru',
        )

        super().__init__()
        self.encoder_feat = seq_encoder_feat
        self.encoder_tgt = seq_encoder_tgt
        self.encoder_geo = seq_encoder_geo

        # self.dial = nn.LSTM(769, 128, batch_first=True, bidirectional = True, dropout = 0.1)

        self.dial = torch.nn.Sequential(
            torch.nn.Linear(769, 64)
        )
        self.head = nn.Linear(self.encoder_feat.embedding_size + self.encoder_tgt.embedding_size  + self.encoder_geo.embedding_size + 64, num_labels)
        self.dropout = nn.Dropout(0.15)
        self.relu = nn.ReLU()
        self.batchnorm = nn.LayerNorm(self.encoder_feat.embedding_size + self.encoder_tgt.embedding_size  + self.encoder_geo.embedding_size  + 64)

    def forward(self, x1, x2, x3, x4):

        # x3, (_, _) = self.dial(x3)
        # x3 = x3.mean(1)
        x3 = self.dial(x3)
        x1 = self.encoder_feat(x1)
        x2 = self.encoder_tgt(x2)
        x4 = self.encoder_geo(x4)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.head(x)


        return x