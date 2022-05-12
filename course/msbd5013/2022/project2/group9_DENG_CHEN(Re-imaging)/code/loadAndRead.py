import pickle
'''
存读pkl文件
'''

def write(lis,fileName):   #write to pkl files
    with open(fileName,'wb') as f:
        pickle.dump(lis,f)

def read(fileName):       # read from pkl files
    with open(fileName,'rb') as f:
        return pickle.load(f)