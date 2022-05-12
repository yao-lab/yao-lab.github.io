import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import scipy.stats  as stats
pd.options.display.max_columns = 50
from utils.reduce_mem import reduce_mem

qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])
cols = [f"F{i}" for i in range(1, 29)]

def get_ratios(coef=0.15):
    qs2 = np.log(qs/(1-qs))*coef
    ratios = stats.norm.cdf(qs2)
    ratios /= ratios[4]
    ratios = pd.Series(ratios, index=qs)
    return ratios.round(3)

#coef between 0.05 and 0.24 is used, probably suboptimal values for now
level_coef_dict = {"id": get_ratios(coef=0.3), "item_id": get_ratios(coef=0.15),
                   "dept_id": get_ratios(coef=0.08), "cat_id": get_ratios(coef=0.07),
                   "store_id": get_ratios(coef=0.08), "state_id": get_ratios(coef=0.07), "_all_": get_ratios(coef=0.05),
                   ("state_id", "item_id"): get_ratios(coef=0.19),  ("state_id", "dept_id"): get_ratios(coef=0.1),
                    ("store_id","dept_id") : get_ratios(coef=0.11), ("state_id", "cat_id"): get_ratios(coef=0.08),
                    ("store_id","cat_id"): get_ratios(coef=0.1)
                  }

def quantile_coefs(q, level):
    ratios = level_coef_dict[level]
    return ratios.loc[q].values

def get_group_preds(pred, level):
    df = pred.groupby(level)[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= quantile_coefs(q, level)[:, None]
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_evaluation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_evaluation', '')}_{q:.3f}_evaluation" for lev, q in zip(df[level].values, q)]
    df = df[["id"]+list(cols)]
    return df

def get_couple_group_preds(pred, level1, level2):
    df = pred.groupby([level1, level2])[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= quantile_coefs(q, (level1, level2))[:, None]
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_evaluation" for lev1,lev2, q in 
                zip(df[level1].values,df[level2].values, q)]
    df = df[["id"]+list(cols)]
    return df



def accuracy2uncertainty(best, sales):

    sub = best.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")
    sub["_all_"] = "Total"
    
    levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
    couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                                ("state_id", "cat_id"),("store_id","cat_id")]
    

    df = []
    for level in levels :
        df.append(get_group_preds(sub, level))
    for level1,level2 in couples:
        df.append(get_couple_group_preds(sub, level1, level2))
    df = pd.concat(df, axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df,df] , axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                        "_evaluation$", "_validation")
    df =reduce_mem(df)
    df.to_csv("submission_uncertainty1.csv.gz", index = False, compression='gzip')