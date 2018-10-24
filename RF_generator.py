import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

wkdir = "D://DeepLearning/"
os.chdir(wkdir)

#处理数据集并且使他们数据顺序相同
print("Preprocessing..."),
data_import = pd.read_table("trainset_gcrma.txt", sep="\t", header="infer", na_values=np.NaN)
clin_data_import = pd.read_table("trainset_clindetails.txt", sep="\t", header="infer", na_values=np.NaN)

clin_data_order = clin_data_import.sort_values(by="GEO asscession number")
clindata = clin_data_order.reset_index(drop=True).T#重新排序并设置索引

data = data_import.iloc[:,range(3,289)].T
data.columns = data_import["symbol"]#重新排序数据集，并且设置表头为基因标记
print("Done")

def un_log2(x):
    return (2 ** x)

#判断是否占样本总量p的基因数据的原始强度都高于A
def pOverA(series, p, A):
    cnt = 0
    for num in series:
        if num > A:
            cnt += 1
    if cnt/series.count() > p:
        del num, cnt
        return True
    del num, cnt
    return False
    
#基因过滤器，过滤掉基因表达变异系数不在0.7到10之间或者不是20％的样品具有了大于100的原始强度的基因
def genefilter(dataframe, syl):
    for i in syl:
        gene = dataframe[i].map(un_log2)#对基因数据的每个值计算unlog2，即作为2的指数乘方
        cv = gene.std()/gene.mean()
        if pOverA(gene, 0.2, 100):#先判断强度，再判断变异系数，加快算法速度
            if cv > 0.7 and cv < 10:
                continue
            else:
                dataframe.pop(i)#删除不符合的基因数据列
        else:
            dataframe.pop(i)#删除不符合的基因数据列
    return dataframe

print("Filtering the genes...")
data = genefilter(data, data_import["symbol"])#过滤数据并生成新的dataframe
print("Done")
fdata = data.values
target = clindata.T[["relapse (1=True)"]]
target = target.values.reshape(286,).astype("int")

print("Fitting in the random forest...")
RF = RandomForestClassifier(n_estimators=10001)#生成随机森林对象
RF.fit(fdata, target)#对数据进行拟合，取奇数个分类器可防止拟合时被随机打破关系
print("Done")

#保存随机森林分类器
with open("RF_model",mode="wb") as f:
    pickle.dump(RF,f)
