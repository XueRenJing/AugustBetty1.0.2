import pandas as pd
import os 
from .ChiMergeGroup import *
import numpy as np


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
def chimerge_clus(tmp1, col, y, max_interval,dropvalue_list=[]):
    tmp1=tmp1.drop(dropvalue_list,axis=1)
    tmp1['status']= y
    print("{} is in processing".format(col))
    if -1 not in set(tmp1[col]):   
        #－1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
        cutOff = ChiMerge(tmp1, col, 'status', max_interval=max_interval,special_attribute=[],minBinPcnt=0)
        tmp1[col+'_Bin'] = tmp1[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[]))
        monotone = BadRateMonotone(tmp1, col+'_Bin', 'status')   # 检验分箱后的单调性是否满足
        while(not monotone):
            # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
            max_interval -= 1
            cutOff = ChiMerge(tmp1, col, 'status', max_interval=max_interval, special_attribute=[],
                                            minBinPcnt=0)
            tmp1[col + '_Bin'] = tmp1[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
            if max_interval == 2:
                # 当分箱数为2时，必然单调
                break
            monotone = BadRateMonotone(tmp1, col + '_Bin', 'status')
        newVar = col + '_Bin'
        tmp1[newVar] = tmp1[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
    else:
        max_interval = 5
        # 如果有－1，则除去－1后，其他取值参与分箱
        cutOff = ChiMerge(tmp1, col, 'status', max_interval=max_interval, special_attribute=[-1],
                                        minBinPcnt=0)
        tmp1[col + '_Bin'] = tmp1[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
        monotone = BadRateMonotone(tmp1, col + '_Bin', 'status',['Bin -1'])
        while (not monotone):
            max_interval -= 1
            # 如果有－1，－1的bad rate不参与单调性检验
            cutOff = sf.ChiMerge(tmp1, col, 'status', max_interval=max_interval, special_attribute=[-1],minBinPcnt=0)
            tmp1[col + '_Bin'] = tmp1[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
            if max_interval == 3:
                # 当分箱数为3-1=2时，必然单调
                break
            monotone = BadRateMonotone(tmp1, col + '_Bin', 'status',['Bin -1'])
            newVar = col + '_Bin'
            tmp1[newVar] = tmp1[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
    return tmp1

def group_class(initial_group_num,num_bins):
    
    for i in range(initial_group_num):
        #如果第一个组没有包含正样本或负样本，向后合并
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2]+num_bins[1][2],
                num_bins[0][3]+num_bins[1][3])]
            continue
    
        """
        如果原本的第一组和第二组都没有包含正样本，或者都没有包含负样本，那即便合并之后，第一行的组也还是没有
        包含两种样本
        所以我们在每次合并完毕之后，还需要再检查，第一组是否已经包含了两种样本
        这里使用continue跳出了本次循环，开始下一次循环，所以回到了最开始的for i in range(20), 让i+1
        这就跳过了下面的代码，又从头开始检查，第一组是否包含了两种样本
        如果第一组中依然没有包含两种样本，则if通过，继续合并，每合并一次就会循环检查一次，最多合并20次
        如果第一组中已经包含两种样本，则if不通过，就开始执行下面的代码
        """
        #已经确认第一组中肯定包含两种样本了，如果其他组没有包含两种样本，就向前合并
        #此时的num_bins已经被上面的代码处理过，可能被合并过，也可能没有被合并
        #但无论如何，我们要在num_bins中遍历，所以写成in range(len(num_bins))
        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i-1:i+1] = [(
                    num_bins[i-1][0],
                    num_bins[i][1],
                    num_bins[i-1][2]+num_bins[i][2],
                    num_bins[i-1][3]+num_bins[i][3])]
            break
            #如果对第一组和对后面所有组的判断中，都没有进入if去合并，则提前结束所有的循环
        else:
            break
        return num_bins
def num_group_chi(data_total,y,initial_group_num,group_num,value_list=[]):
    var_bin_list=[]
    var_dict_total={}
    iv_dict={}
    for col in value_list:
        data_total[col]=data_total[col].fillna(-998)
        data_total["qcut"], updown = pd.qcut(data_total[col], retbins=True, q=int(initial_group_num),duplicates="drop")#等频分箱
        coount_y0 = data_total[data_total[y] == 0].groupby(by="qcut").count()[y]
        coount_y1 = data_total[data_total[y] == 1].groupby(by="qcut").count()[y]
        num_bins = [*zip(updown,updown[1:],coount_y0,coount_y1)]
        num_bins=group_class(initial_group_num,num_bins)
        gg=list([k[1] for k in num_bins])
        gg.append(-9999.0)
        gg.sort()
        data_total[col+"_group"]=pd.cut(data_total[col], gg)
        data_total=data_total.sort_values(by=col)
        data=data_total.drop_duplicates([col+"_group"])
        i=1
        dict_total={}
        for k in data[col+"_group"]:
            dict1={str(k):i}
            dict_total=dict(dict_total,**dict1)
            i=i+1  
        data_total[col+"_group"]=data_total[col+"_group"].map(lambda x :dict_total.get(str(x),0))
        data_total=chimerge_clus(data_total, col+"_group", data_total[y], max_interval=group_num,dropvalue_list=[])
        total_dict={}
        list_group=list(set(data_total[col+"_group_Bin"]))
        list_group.sort()
        for m in range(0,len(list_group)):
            h="Bin "+str(m)
            if h=="Bin 0":
                d_tatal=data_total[data_total[col+"_group_Bin"]==h]
                v_dict={h:[-9999.0,max(d_tatal[col])]}
            elif h=="Bin "+str(len(list_group)-1):
                d_tatal=data_total[data_total[col+"_group_Bin"]==h]
                v_dict={h:[min_1,10000000000]}                
            else:
                d_tatal=data_total[data_total[col+"_group_Bin"]==h]
                v_dict={h:[min_1,max(d_tatal[col])]}
            min_1=max(d_tatal[col])
            total_dict=dict(total_dict,**v_dict)
        # print(total_dict)
        v_total_dict={col:{"bin":total_dict,"value_name":col+"_group_Bin"}}
        var_dict_total=dict(var_dict_total,**v_total_dict)
        idict={col+"_group_Bin":caliv(data_total,col+"_group_Bin",y)} 
        iv_dict=dict(iv_dict,**idict)
        var_bin_list.append(col+"_group_Bin")
    return {"data":data_total,"var_dict_total":var_dict_total,"var_bin_list":var_bin_list,"iv_dict":iv_dict}

def num_group_chi_main(train_data,id_name,y,initial_group_num,group_num): 
    columns_data=train_data.dtypes.reset_index(drop=False)
    columns_list=list(columns_data[(columns_data[0]=="int64")|(columns_data[0]=="float64")|(columns_data[0]=="int32")|(columns_data[0]=="float32")]["index"])
    columns_list.remove(id_name)
    columns_list.remove(y)
    num_value_bin=[]
    num_value=[]
    for k in columns_list:
        if len(train_data[k].unique())>5:
            num_value_bin.append(k)
        else:
            num_value.append(k)
    gg=num_group_chi(train_data,"isDefault",20,5,value_list=num_value_bin)
    iv_dict={}
    for col in num_value:
        idict={col:caliv(train_data,col,y)} 
        iv_dict=dict(iv_dict,**idict)
    var_bin_list=gg.get("var_bin_list")
    for m in num_value:
        # print(m)
        var_bin_list.append(m)
    iv_dict=dict(iv_dict,**gg.get("iv_dict"))
    if id_name in var_bin_list:
        var_bin_list.remove(id_name)
    return {"data":gg.get("data"),"var_dict_total":gg.get("var_dict_total"),"var_bin_list":var_bin_list,"iv_dict":iv_dict}
if __name__ == '__main__':
    train_data = pd.read_csv(r"F:\论文-风控\train.csv")  #读取训练集
    gg=num_group_chi_main(train_data,"id","isDefault",20,5)






