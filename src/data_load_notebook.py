import pandas as pd
import os

paths = ['..\data\S1_bkg_mapA_11x11.txt',
         '..\data\S1_mapA_11x11.txt',
         '..\data\S2_bkg_mapA_11x11.txt',
         '..\data\S2_mapA_11x11.txt']
#abspaths=[]
#for i in paths:
#    abspaths.append(os.path.abspath(i))

# nomi standard per campioni di questo tipo, per molte cose è importante che siano così [non flessibile]
names_col = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]

data1bkg = pd.read_csv(paths[1], names=names_col, delim_whitespace=True)
data1 = pd.read_csv(paths[0], names=names_col, delim_whitespace=True)
data2bkg = pd.read_csv(paths[3], names=names_col, delim_whitespace=True)
data2 = pd.read_csv(paths[2], names=names_col, delim_whitespace=True)


iterpath = os.scandir('..\data\Database_Raman')
database = dict()
for i in iterpath:
    if (i.name != 'RRUFF_list.txt') & (i.name != 'BANK_LIST.txt') :
    # print(i.name)
        database[f'{i.name[:-4]}'] = pd.read_csv(f'..\data\Database_Raman' +f'\{i.name}', names=['K','H'], delim_whitespace=True )
