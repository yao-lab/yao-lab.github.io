import numpy as np
import pandas as pd
import scipy.stats  as stats
from collections import deque
import time
from tqdm import tqdm

# reduce memory
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

# name: name of csv.gz

def naive_forecast(prediction, name):
    def get_couple_group_preds_val(pred, level1, level2):
        df = pred.groupby([level1, level2])[cols].sum()
        q = np.repeat(qs, len(df))
        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        y_hat = df.iloc[:,2:30].to_numpy()
        y = df.iloc[:,30:].to_numpy()
        sum_errs = np.sum((y - y_hat)**2, axis=1) # the residuals
        stdev = np.sqrt(1/(len(y)-2) * sum_errs) # the sd of residuals
        df[pred_cols] += (ratios[q]*stdev)[:, None]
        df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in
                    zip(df[level1].values,df[level2].values, q)]
        df = df[["id"]+list(cols)]
        return df.iloc[:,:29]
    
    def get_group_preds_val(pred, level):
        df = pred.groupby(level)[cols].sum()
        q = np.repeat(qs, len(df))
        df = pd.concat([df]*9, axis=0, sort=False)
        df.reset_index(inplace = True)
        y_hat = df.iloc[:,1:29].to_numpy()
        y = df.iloc[:,29:].to_numpy()
        sum_errs = np.sum((y - y_hat)**2, axis=1) # the residuals
        stdev = np.sqrt(1/(len(y)-2) * sum_errs) # the sd of residuals
        df[pred_cols] += (ratios[q]*stdev)[:, None]
        if level != "id":
            df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        else:
            df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
        df = df[["id"]+list(cols)]

        return df.iloc[:,:29]

    y_hat_val = prediction.iloc[:len(prediction)//2,:] # y_hat
    y_val = pd.read_csv('./data/sales_train_evaluation.csv').iloc[:,1919:] # the y label
    df_y_val = pd.concat([y_hat_val, y_val], axis=1)
    y_hat_eval = prediction.iloc[len(prediction)//2:,:]

    cols = df_y_val.iloc[:,1:].columns
    pred_cols = df_y_val.iloc[:,1:29].columns
    qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])

    dq = deque()
    dq.append(0.0)
    for q in [0.5, 0.67, 0.95, 0.99]:
        lower, upper = stats.norm.interval(q)
        dq.appendleft(lower)
        dq.append(upper)

    ratios = pd.Series(dq, index=qs).round(3)
    
    sales_val = pd.read_csv("./data/sales_train_validation.csv")
    sub_val = df_y_val.merge(sales_val[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")
    sub_val["_all_"] = "Total"

    sales_eval = pd.read_csv("./data/sales_train_evaluation.csv")
    sub_eval = y_hat_eval.merge(sales_eval[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")
    sub_eval["_all_"] = "Total"

    levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
    couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                                ("state_id", "cat_id"),("store_id","cat_id")]
    
    df_val = []
    for level in levels :
        df_val.append(get_group_preds_val(sub_val, level))
        #df_eval.append(get_group_preds_eval(sub_eval, level, stdev))
    for level1,level2 in couples:
        df_val.append(get_couple_group_preds_val(sub_val, level1, level2))
        #df_eval.append(get_couple_group_preds_eval(sub_eval, level1, level2, stdev))
    df_val = pd.concat(df_val, axis=0, sort=False)
    df_val.reset_index(drop=True, inplace=True)
    
    df = pd.concat([df_val, df_val] , axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                            "_validation", "_evaluation")
    df =reduce_mem(df)
    df.to_csv(name + ".csv.gz", index = False, compression='gzip')
    

if __name__ == "__main__":
    ensembel_result = pd.read_csv('submission_accuray3.csv.gz', compression='gzip')
    naive_forecast(ensembel_result, 'temp')