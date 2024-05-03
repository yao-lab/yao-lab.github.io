def split(test_start, df_date):
    test_end = test_start + 100
    val_start = test_start - 1200
    train_ind = df_date < val_start
    val_ind = (df_date >= val_start) & (df_date < test_start)
    test_ind = (df_date >= test_start) & (df_date < test_end)
    return train_ind, val_ind, test_ind

def R2_OOS(y, y_pred):
    R = 1 - sum( (y - y_pred) ** 2 ) / sum( y ** 2 )
    return R
