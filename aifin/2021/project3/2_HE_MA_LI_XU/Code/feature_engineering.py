# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:00:11 2021

@author: xtcxxx
"""


#calendar and merge
import numpy as np
import pandas as pd 
calendar_=pd.read_csv("calendar_new.csv")
calendar_=calendar_.iloc[:,1:16]
from tqdm import tqdm
from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder() 
category=['event_name_1','event_type_1','event_name_2','event_type_2']
for i in tqdm(category):
  calendar_[i+'_']=labelencoder.fit_transform(calendar_[i])
calendar_=calendar_.drop(['event_name_1','event_type_1','event_name_2','event_type_2'],axis=1)
calendar_.to_csv("calendar_new1.csv")
sales_train_evaluation_=pd.read_csv("sales_train_evaluation.csv")
sell_prices_=pd.read_csv("sell_prices.csv")
sales=pd.melt(sales_train_evaluation_,id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],
              var_name='d',value_name='demand')
sales=pd.merge(sales,calendar_,on='d',how='left')
sales=pd.merge(sales,sell_prices_,on=['item_id','store_id','wm_yr_wk'],how='left')
sales['sell_price']=sales['sell_price'].fillna(sales.groupby('id')['sell_price'].transform('mean'))
labelencoder=preprocessing.LabelEncoder() 
category=['state_id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])
sales=sales.drop(["state_id"],axis=1)
sales.to_csv("data_newww.csv")

import numpy as np
import pandas as pd 
sales_train_evaluation_=pd.read_csv("sales_train_evaluation.csv")
sell_prices_=pd.read_csv("sell_prices.csv")
from tqdm import tqdm
from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder() 
category=['id','item_id','dept_id','cat_id','store_id','state_id']
for i in tqdm(category):
  sales_train_evaluation_[i+'_']=labelencoder.fit_transform(sales_train_evaluation_[i])
sales=pd.melt(sales_train_evaluation_,id_vars=['id','item_id','dept_id','cat_id','store_id','state_id',
                                               'id_','item_id_','dept_id_','cat_id_','store_id_','state_id_'],
              var_name='d',value_name='demand')
import numpy as np
import pandas as pd 
calendar_=pd.read_csv("calendar_new1.csv")
sale=pd.read_csv("sales_newwww.csv")
sale=pd.merge(sale,calendar_,on='d',how='left')
sale=pd.merge(sale,sell_prices_,on=['item_id','store_id','wm_yr_wk'],how='left')
sales.to_csv("sales_newwww.csv")


'''
#LabelEncoder
import numpy as np
import pandas as pd 
sales=pd.read_csv("sales_new.csv")

from tqdm import tqdm
from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder() 
category=['id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])
from tqdm import tqdm
from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder() 
category=['item_id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])
sales.drop('item_id',axis = 1,inplace = True)
labelencoder=preprocessing.LabelEncoder() 
category=['dept_id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])
sales.drop('dept_id',axis = 1,inplace = True)
labelencoder=preprocessing.LabelEncoder() 
category=['cat_id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])
sales.drop('cat_id',axis = 1,inplace = True)

sales=sales.iloc[:,1:25]
from tqdm import tqdm
from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder() 
category=['store_id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])

sales.drop('store_id',axis = 1,inplace = True)
labelencoder=preprocessing.LabelEncoder() 
category=['state_id']
for i in tqdm(category):
  sales[i+'_']=labelencoder.fit_transform(sales[i])

sales.drop('state_id',axis = 1,inplace = True)

sales.drop('date',axis = 1,inplace = True)
sales.to_csv("data_final_c.csv")
sales.drop(['id_','dept_id_','cat_id_','store_id_'],axis = 1,inplace = True)
sales.to_csv("data_final_s.csv",index=False)
'''

'''
#plot features
import pandas as pd

sales = pd.read_csv('drive/My Drive/Colab Notebooks/sales_train_evaluation.csv')
prices = pd.read_csv('drive/My Drive/Colab Notebooks/sell_prices.csv')
cal = pd.read_csv('drive/My Drive/Colab Notebooks/calendar.csv')
sales_d = sales.drop(['state_id','store_id','cat_id','dept_id','item_id'], axis=1)
sales_p = sales_d.set_index('id').T.merge(cal.set_index('d')['date'],
                                                 left_index=True,
                                                 right_index=True,
                                                  validate='1:1') 
sales_p = sales_p.set_index('date')
stores = sorted(set(prices['store_id']))
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
name=["Rolling Average Sales vs. Time (per store) with window size:3",
      "Rolling Average Sales vs. Time (per store) with window size:7",
      "Rolling Average Sales vs. Time (per store) with window size:14",
      "Rolling Average Sales vs. Time (per store) with window size:21",
      "Rolling Average Sales vs. Time (per store) with window size:28"]

window_size=[3,7,14,21,28]
for i,j in enumerate(window_size):

 fig = go.Figure()

 for store in stores:
  store_items=[]
  for a in sales_p.columns:
    if store in a:
      store_items.append(a)

  data = sales_p[store_items].sum(axis=1)
  data = data.rolling(j).mean()
  l=np.arange(len(data))
  fig.add_trace(go.Scatter(x=l, y=data, name=store))
  
 fig.update_layout(xaxis_title="Time", yaxis_title="Sales",title=name[i],title_x=0.5)
 fig.show()
 import plotly.express as px
import numpy as np 
gp = sales.groupby(['state_id','store_id','cat_id','dept_id'],as_index=False)
gp = gp['item_id'].count().dropna()

fig = px.treemap(gp, path=['state_id', 'store_id','cat_id', 'dept_id', ], values='item_id',
                  color='item_id',
                  color_continuous_scale = 'RdBu')
            
fig.update_layout(treemapcolorway = ["red", "blue"])
fig.show()
import plotly.express as px
import numpy as np 
px.bar(df, x="WI", y="Average Sales", color="WI", title="Sales vs snap")
'''