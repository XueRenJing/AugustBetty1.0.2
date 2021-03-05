# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:22:08 2021

@author: 560350
"""
from scipy.stats import scoreatpercentile
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from .NumBestGroup import *
from .CharBestGroup import *
import statsmodels.api as sm
class MakeModel(object):
    def __init__(self,data,id_name,y_name,iv_num,group_num,drop_value=[]):
        self.train_data,self.val_data=train_test_split(data, test_size=0.3, random_state=545)
        self.id_name=id_name
        self.y_name=y_name
        self.drop_value=drop_value
        self.iv_num=iv_num
        self.group_num=group_num
    
    
    
    def select_value_iv(self):
        '''
        返回数值变量 字符变量中大于设定的iv_num的变量列表  
        '''
        feature_list = [k for k,v in self.num_object.get("iv_dict").items() if v.get("iv")>self.iv_num ]
        feature_list1 = [k for k,v in self.char_object.get("iv_dict").items() if v.get("iv")>self.iv_num ]
        for j in feature_list1:
            feature_list.append(j)
        self.feature_list=feature_list
    
    def corr_x(self):
        return X.corr()
    
    def ks_calc_auc(self,data,pred,y_label):
      '''
      功能: 计算KS值，输出对应分割点和累计分布函数曲线图
      输入值:
      data: 二维数组或dataframe，包括模型得分和真实的标签
      pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
      y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
      输出值:
      'ks': KS值
      '''
      fpr,tpr,thresholds= roc_curve(data[y_label],data[pred])
      ks = max(tpr-fpr)
      return ks
    

    
    def bin_woe(self,n,x):   
        var=n.replace("_group_Bin","")
        for b in self.num_object.get("var_dict_total").get(var).get("bin"):
            # print(b)
            if x>self.num_object.get("var_dict_total").get(var).get("bin").get(b)[0]  and x<=self.num_object.get("var_dict_total").get(var).get("bin").get(b)[1]:
                return self.num_object.get("iv_dict").get(n).get("woe").get(b)
            else:
                continue  
            
    def deal_raw_data(self,data):
        for n in self.num_object.get("var_bin_list"):
            print(n)
            if "_group_Bin" in n:
                var=n.replace("_group_Bin","")
                data[var]=data[var].fillna(-998)
                data[self.num_object.get("var_dict_total").get(var).get("value_name")]= data[var].map(lambda x :self.bin_woe(n,x))
            else:
                data[n]= data[n].map(lambda x :self.num_object.get("iv_dict").get(n).get("woe").get(x))
        for n in self.char_object.get("var_bin_list"):
            if "_group_Bin" in n:
                n=n.replace("_group_Bin","")
                data[n]=data[n].fillna("null")
                b=self.char_object.get("var_dict_total").get(n).get("var_name")
                data[self.char_object.get("var_dict_total").get(n).get("var_name")]= data[n].map(lambda x :self.char_object.get("iv_dict").get(b).get("woe").get(self.char_object.get("var_dict_total").get(n).get("bin").get(x)))
            else:
                data[n]= data[n].map(lambda x :self.char_object.get("iv_dict").get(n).get("woe").get(x))
        return data

    def model_main(self,feature_list=[]):     
        self.num_object=num_group_chi_main(self.train_data,self.id_name,self.y_name,20,self.group_num)
        self.char_object=char_group_chi_main(self.train_data,self.id_name,self.y_name,5,self.drop_value)
        # num_object_data=self.woe_change(self.num_object)
        # char_object_data=self.woe_change(self.char_object)   
        self.select_value_iv()
        if len(feature_list)==0:
            feature_list=self.feature_list
        else:
            feature_list=feature_list
        # 处理原始训练集
        self.train_data=self.deal_raw_data(self.train_data)
        y = self.train_data[self.y_name]
        X = self.train_data[feature_list]
        X['intercept'] = [1]*X.shape[0]
        LR = sm.Logit(y, X).fit()
        summary = LR.summary()
        pvals = LR.pvalues
        pvals = pvals.to_dict()
        self.train_data['predicted'] = LR.predict(X.astype(float))
        # 处理原始测试集
        self.val_data=self.deal_raw_data(self.val_data)
        X_val = self.val_data[self.feature_list]
        X_val['intercept'] = [1]*X_val.shape[0]
        self.val_data['predicted']= LR.predict(X_val.astype(float))
        ks=self.ks_calc_auc(self.train_data,'predicted', self.y_name)
        auc=roc_auc_score(list(self.train_data[self.y_name]),list(self.train_data['predicted']))
        val_ks=self.ks_calc_auc(self.val_data,'predicted', self.y_name)
        val_auc=roc_auc_score(list(self.val_data[self.y_name]),list(self.val_data['predicted']))        
        self.result={"model_summary":summary,"ks":ks,"auc":auc,"val_ks":val_ks,"val_auc":val_auc}        
if __name__ == '__main__':
    data = pd.read_csv(r"F:\论文-风控\train.csv")  #读取训练集
    m=MakeModel(data,"id","isDefault",0.02,5,drop_value=["issueDate","earliesCreditLine"])
    m.model_main()
    print(m)

    
    
    
    

