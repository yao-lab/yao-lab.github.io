import numpy as np
import pandas as pd
import gc


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print(
        "Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem)
    )

    return df


def preprocess_sales(sales, start=1400, upper=1970):
    if start is not None:
        print("dropping...")
        to_drop = [f"d_{i+1}" for i in range(start - 1)]
        print(sales.shape)
        sales.drop(to_drop, axis=1, inplace=True)
        print(sales.shape)
    # =======
    print("adding...")
    new_columns = ["d_%i" % i for i in range(1942, upper, 1)]
    for col in new_columns:
        sales[col] = np.nan
    print("melting...")
    sales = sales.melt(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "scale1",
            "sales1",
            "start",
            "sales2",
            "scale2",
        ],
        var_name="d",
        value_name="demand",
    )
    #
    return sales


def preprocess_calendar(calendar):
    global maps, mods
    calendar["event_name"] = calendar["event_name_1"]
    calendar["event_type"] = calendar["event_type_1"]

    map1 = {mod: i for i, mod in enumerate(calendar["event_name"].unique())}
    calendar["event_name"] = calendar["event_name"].map(map1)
    map2 = {mod: i for i, mod in enumerate(calendar["event_type"].unique())}
    calendar["event_type"] = calendar["event_type"].map(map2)
    calendar["nday"] = calendar["date"].str[-2:].astype(int)
    maps["event_name"] = map1
    maps["event_type"] = map2
    mods["event_name"] = len(map1)
    mods["event_type"] = len(map2)
    calendar["wday"] -= 1
    calendar["month"] -= 1
    calendar["year"] -= 2011
    mods["month"] = 12
    mods["year"] = 6
    mods["wday"] = 7
    mods["snap_CA"] = 2
    mods["snap_TX"] = 2
    mods["snap_WI"] = 2

    calendar["nb"] = calendar.index + 1

    calendar.drop(
        [
            "event_name_1",
            "event_name_2",
            "event_type_1",
            "event_type_2",
            "date",
            "weekday",
        ],
        axis=1,
        inplace=True,
    )
    return calendar


def make_dataset(categorize=False, start=1400, upper=1970):
    global maps, mods
    print("loading calendar...")
    calendar = pd.read_csv("../data/calendar.csv")
    print("loading sales...")
    sales = pd.read_csv("../data/sales_train_validation.csv")
    cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    if categorize:
        for col in cols:
            temp_dct = {mod: i for i, mod in enumerate(sales[col].unique())}
            mods[col] = len(temp_dct)
            maps[col] = temp_dct
        for col in cols:
            sales[col] = sales[col].map(maps[col])
        #

    sales = preprocess_sales(sales, start=start, upper=upper)
    calendar = preprocess_calendar(calendar)
    calendar = reduce_mem_usage(calendar)
    print("merge with calendar...")
    sales = sales.merge(calendar, on="d", how="left")
    # del calendar

    print("reordering...")
    sales.sort_values(by=["id", "nb"], inplace=True)
    print("re-indexing..")
    sales.reset_index(inplace=True, drop=True)
    gc.collect()

    sales["n_week"] = (sales["nb"] - 1) // 7
    sales["nday"] -= 1
    mods["nday"] = 31
    sales = reduce_mem_usage(sales)
    calendar = calendar.loc[calendar.nb >= start]
    gc.collect()
    return sales, calendar
