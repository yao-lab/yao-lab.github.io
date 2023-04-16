import pandas as pd
import numpy as np

Folder = 'E:\\HKUST\\CSIC5011\\Mini_project\\Source_data\\'
SNP_CSV_Name = 'ceph_hgdp_minor_code_XNA.betterAnnotated.csv'
Nation_CSV_Name = 'ceph_hgdp_minor_code_XNA.sampleInformation.csv'

SNP_Dataframe = pd.read_csv(Folder + SNP_CSV_Name)
Nation_Dataframe = pd.read_csv(Folder + Nation_CSV_Name)

SNP_Raw = (SNP_Dataframe.iloc[1:-1, 3:-1]).to_numpy()
Nation_Raw = (Nation_Dataframe.iloc[:, 5]).to_numpy()

np.save('SNP_Raw', SNP_Raw)
np.save('Nation_Raw', Nation_Raw)
