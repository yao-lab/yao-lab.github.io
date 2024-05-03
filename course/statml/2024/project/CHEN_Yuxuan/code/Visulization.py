import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
import warnings

warnings.filterwarnings("ignore")


OLSH_R2 = pd.read_csv('OLSH_R2.csv')
OLS3_R2 = pd.read_csv('OLS3_R2.csv')
OLS3H_R2 = pd.read_csv('OLS3H_R2.csv')
LASSOH_R2 = pd.read_csv('LASSOH_R2.csv')
RIDGEH_R2 = pd.read_csv('RIDGEH_R2.csv')
ENETH_R2 = pd.read_csv('ENETH_R2.csv')
PCR_R2 = pd.read_csv('PCR_R2.csv')
PLS_R2 = pd.read_csv('PLS_R2.csv')
GBRH_R2 = pd.read_csv('GBRH_R2.csv')
RF_R2 = pd.read_csv('RF_R2.csv')

# Table 1
t1 = {'OLS + H': [np.mean(OLSH_R2[10:])], 
      'OLS-3': [np.mean(OLS3_R2[10:])],
      'OLS-3 + H': [np.mean(OLS3H_R2[10:])],
      'LASSO + H': [np.mean(LASSOH_R2[10:])],
      'Ridge + H': [np.mean(RIDGEH_R2[10:])],
      'ENET + H': [np.mean(ENETH_R2[10:])],
      'PCR': [np.mean(PCR_R2[10:])],
      'PLS': [np.mean(PLS_R2[10:])],
      'GBDT + H': [np.mean(GBRH_R2[10:])],
      'RF': [np.mean(RF_R2[10:])]}

table1 = pd.DataFrame(t1)
table1.index = ['']
print(table1)

# Figure 1
n = len(OLS3_R2)
name = ['OLS-3 + H', 'LASSO + H', 'Ridge + H', 'ENET + H', 'PCR', 'PLS', 'GBDT + H', 'RF']
R2_0 = [OLS3H_R2, LASSOH_R2, RIDGEH_R2, ENETH_R2, PCR_R2, PLS_R2, GBRH_R2, RF_R2]
R2 = []
for i in range(n):
    R2_i = np.zeros(len(name))
    for j in range(len(name)):
        R2_i[j] = R2_0[j].loc[i]
    R2.append(R2_i)
    
plt.figure(figsize=(12, 8))
plt.plot(1987 + np.array(range(30)), R2)
plt.xlabel('Year')
plt.ylabel('R2 Score')
plt.ylim(-0.3, 0.2)
plt.legend(name, loc='upper right')
plt.savefig('Figure1.png')








# Figure 2
LASSOH_comp = pd.read_csv('LASSOH_comp.csv')
ENETH_comp = pd.read_csv('ENETH_comp.csv')
PCR_comp = pd.read_csv('PCR_comp.csv')
PLS_comp = pd.read_csv('PLS_comp.csv')

year = 1987 + np.array(range(30))

plt.figure(figsize=(12, 12.5))
plt.subplot(4, 1, 1)
plt.plot(year, LASSOH_comp)
plt.title('LASSO + H', loc = 'left')
plt.xlabel('Year')
plt.ylabel('# of Char.')
plt.subplot(4, 1, 2)
plt.plot(year, ENETH_comp)
plt.title('ENET + H', loc = 'left')
plt.xlabel('Year')
plt.ylabel('# of Char.')
plt.subplot(4, 1, 3)
plt.plot(year, PCR_comp)
plt.title('PCR', loc = 'left')
plt.xlabel('Year')
plt.ylabel('# of Comp.')
plt.subplot(4, 1, 4)
plt.plot(year, PLS_comp)
plt.title('PLS', loc = 'left')
plt.xlabel('Year')
plt.ylabel('# of Comp.')
plt.savefig('Figure2.png')








# Figure 3
LASSOH_Importance = pd.read_csv('LASSOH_Importance.csv')
RF_Importance = pd.read_csv('RF_Importance.csv')
ENETH_Importance = pd.read_csv('ENETH_Importance.csv')
PCR_Importance = pd.read_csv('PCR_Importance.csv')
PLS_Importance = pd.read_csv('PLS_Importance.csv')
GBRH_Importance = pd.read_csv('GBRH_Importance.csv')

macro = ['dp', 'ep_macro', 'bm_macro', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
im_name = np.array([p for p in LASSOH_Importance.columns.values if p not in macro])

LASSOH_Im = []
RF_Im = []
ENETH_Im = []
PCR_Im = []
PLS_Im = []
GBRH_Im = []
for i in range(len(im_name)):
    LASSOH_Im.append(LASSOH_Importance[im_name[i]].values)
    RF_Im.append(RF_Importance[im_name[i]].values)
    ENETH_Im.append(ENETH_Importance[im_name[i]].values)
    PCR_Im.append(PCR_Importance[im_name[i]].values)
    PLS_Im.append(PLS_Importance[im_name[i]].values)
    GBRH_Im.append(GBRH_Importance[im_name[i]].values)

LASSOH_val = heapq.nlargest(20, LASSOH_Im)
LASSOH_name = im_name[list( map(LASSOH_Im.index, LASSOH_val) )]
LASSOH_val = np.array(LASSOH_val).reshape(-1)

RF_val = heapq.nlargest(20, RF_Im)
RF_name = im_name[list( map(RF_Im.index, RF_val) )]
RF_val = np.array(RF_val).reshape(-1)

ENETH_val = heapq.nlargest(20, ENETH_Im)
ENETH_name = im_name[list( map(ENETH_Im.index, ENETH_val) )]
ENETH_val = np.array(ENETH_val).reshape(-1)

PCR_val = heapq.nlargest(20, PCR_Im)
PCR_name = im_name[list( map(PCR_Im.index, PCR_val) )]
PCR_val = np.array(PCR_val).reshape(-1)

PLS_val = heapq.nlargest(20, PLS_Im)
PLS_name = im_name[list( map(PLS_Im.index, PLS_val) )]
PLS_val = np.array(PLS_val).reshape(-1)

GBRH_nonzero = np.count_nonzero(np.array(GBRH_Im))
GBRH_val = heapq.nlargest(GBRH_nonzero, GBRH_Im)
GBRH_name = im_name[list( map(GBRH_Im.index, GBRH_val) )]
GBRH_val = np.array(GBRH_val).reshape(-1)

plt.figure(figsize=(18, 13))
plt.subplot(3, 2, 1)
plt.barh(LASSOH_name, LASSOH_val)
plt.title('LASSO + H', loc = 'left')
plt.xlabel('Importance')
plt.subplot(3, 2, 5)
plt.barh(RF_name, RF_val)
plt.title('RF', loc = 'left')
plt.xlabel('Importance')
plt.subplot(3, 2, 2)
plt.barh(ENETH_name, ENETH_val)
plt.title('ENET + H', loc = 'left')
plt.xlabel('Importance')
plt.subplot(3, 2, 3)
plt.barh(PCR_name, PCR_val)
plt.title('PCR', loc = 'left')
plt.xlabel('Importance')
plt.subplot(3, 2, 4)
plt.barh(PLS_name, PLS_val)
plt.title('PLS', loc = 'left')
plt.xlabel('Importance')
plt.subplot(3, 2, 6)
plt.barh(GBRH_name, GBRH_val)
plt.title('GBR + H', loc = 'left')
plt.xlabel('Importance')
plt.savefig('Figure3.png')