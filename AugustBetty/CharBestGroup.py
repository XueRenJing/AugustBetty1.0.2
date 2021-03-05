# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:04:14 2021

@author: 560350
"""
import os 
from .ChiMergeGroup import *
import numpy as np

def chimerge_clus(tmp1, col, y, max_interval,dropvalue_list=[]):
    tmp1=tmp1.drop(dropvalue_list,axis=1)
    tmp1['status']= y
    print("{} is in processing".format(col))
    cutOff = ChiMerge(tmp1, col, 'status', max_interval=max_interval,special_attribute=[],minBinPcnt=0)
    tmp1[col+'_Bin'] = tmp1[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[]))
    return tmp1

def caliv(df,string,status):
    eps=0.00001
    crosstab=pd.crosstab(df[string],df[status]) + eps
    sample = df[status].value_counts() + eps
    frequence = crosstab/sample
    frequence['woe'] = np.log(frequence[1] / frequence[0])
    frequence['iv'] = (frequence[1] - frequence[0])*frequence['woe']
    frequence=frequence.reset_index(drop=False)
    var_dict=frequence.set_index(string)["woe"].to_dict()
    return {"iv":frequence['iv'].sum(),"woe":var_dict}

def char_group_chi(data_total,y,group_num,value_list=[]):
    '''
    Parameters
    ----------
    data_total :数据集
    value_list : 需要分组的字符变量名列表
    y : y变量名
    group_num : 分组数
    Returns 加了新的分组变量的数据集，分组映射字典,分组之后的新变量的变量名列表
    -------
    None.

    '''
    var_bin_list=[]
    var_dict_total={}
    iv_dict={}
    for col in value_list:
        data_total[col]=data_total[col].fillna("null")
        data=pd.merge(data_total.groupby([col])[y].sum().reset_index(drop=False),data_total.groupby([col])[y].count().reset_index(drop=False),how='inner',left_on=col,right_on=col)
        data["rate"]=data[y+"_x"]/data[y+"_y"]
        data=data.sort_values(by="rate")
        i=1
        dict_total={}
        for k in data[col]:
            dict1={k:i}
            dict_total=dict(dict_total,**dict1)
            i=i+1   
        data_total[col+"_group"]=data_total[col].map(lambda x :dict_total.get(x,0))

        # iv_dict={}
        # for m in [3,4,5,6,7]:
        #     data_total=chimerge_clus(data_total, col+"_group", train_data[y], max_interval=m,dropvalue_list=[])
        #     print(caliv(data_total,col+"_group_Bin",y))
        #     idict={str(m):caliv(data_total,col+"_group_Bin",y)} 
        #     iv_dict=dict(iv_dict,**idict)
        # print(iv_dict)
        # groupnum = max(iv_dict, key=iv_dict.get)   
        # print(groupnum)  
        data_total=chimerge_clus(data_total, col+"_group", data_total[y], max_interval=int(5),dropvalue_list=[])
        bin_data=data_total.drop_duplicates([col,col+"_group_Bin"],keep="last")[[col,col+"_group_Bin"]]
        var_dict=bin_data.set_index(col)[col+"_group_Bin"].to_dict()
        var_bin_list.append(col+"_group_Bin")
        vdict={col:{"bin":var_dict,"var_name":col+"_group_Bin"}}
        var_dict_total=dict(var_dict_total,**vdict)
        
        idict={col+"_group_Bin":caliv(data_total,col+"_group_Bin",y)} 
        iv_dict=dict(iv_dict,**idict)
    return {"data":data_total,"var_dict_total":var_dict_total,"var_bin_list":var_bin_list,"iv_dict":iv_dict}

def char_group_chi_main(data_total,id_name,y,group_num,drop_value=[]): 
    columns_data=data_total.dtypes.reset_index(drop=False)
    columns_list=list(columns_data[columns_data[0]=="object"]["index"])
    if id_name in columns_list or y in columns_list:
        columns_list.remove(id_name)
        columns_list.remove(y)
    else:
        pass
    for m in drop_value:
        columns_list.remove(m)
    char_value_bin=[]
    char_value=[]
    for k in columns_list:
        if len(data_total[k].unique())>5:
            char_value_bin.append(k)
        else:
            char_value.append(k)
    # print(char_value_bin)
    # print(char_value)
    gg=char_group_chi(data_total,"isDefault",5,value_list=char_value_bin)
    iv_dict={}
    for col in char_value:
        data=data_total[col].drop_duplicates([col])
        for k in data[col]:
            dict1={k:i}
            dict_total=dict(dict_total,**dict1)
            i=i+1   
        data_total[col+"_group"]=data_total[col].map(lambda x :dict_total.get(x,0))
        idict={col:caliv(data_total,col,y)} 
        iv_dict=dict(iv_dict,**idict)
    var_bin_list=gg.get("var_bin_list")
    if len(char_value)>0:
        var_bin_list.append(char_value)
    elif id_name in var_bin_list:
        var_bin_list.remove(id_name)
    else:
        pass
    # print(var_bin_list)
    iv_dict=dict(iv_dict,**gg.get("iv_dict"))
    return {"data":gg.get("data"),"var_dict_total":gg.get("var_dict_total"),"var_bin_list":var_bin_list,"iv_dict":iv_dict}

if __name__ == '__main__':
    train_data = pd.read_csv(r"F:\论文-风控\train.csv")  #读取训练集
    gg=char_group_chi_main(train_data,"id","isDefault",5,drop_value=["issueDate","earliesCreditLine"])
    

    
