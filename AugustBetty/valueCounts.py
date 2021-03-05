import pandas as pd
from tqdm import tqdm
def valueCounts(data,id_name,y_name=None):
    '''
        输入数据以及主键、y值(有就输入，没有只要输入data和id_name就可以)
        -------
        输出两个数据集，一个是字符变量的：
    	value	counts	varname
        0	S	644	Embarked
        1	C	168	Embarked
        2	Q	77	Embarked
        一个是数值型的：
        	        count	   mean	                  std	        min	 25%	 50%	75%	      max
            Pclass	891.0	2.308641975308642	0.8360712409770513	1.0	 2.0    3.0	    3.0	     3.0
            Age	    714.0	29.69911764705882	14.526497332334044	0.42 20.125	28.0	38.0	80.0
            SibSp	891.0	0.5230078563411896	1.1027434322934275	0.0	0.0	    0.0	    1.0	    8.0
            Parch	891.0	0.38159371492704824	0.8060572211299559	0.0	0.0	    0.0	    0.0	    6.0
            Fare	891.0	32.2042079685746	49.693428597180905	0.0	7.9104	14.4542	31.0	512.3292
    '''
    if  y_name==None:
        data_number=pd.DataFrame()
        data_char=pd.DataFrame()
        for k in tqdm(data.drop([id_name],axis=1).columns):
            if "int" in str(data[k].dtypes ) or "float" in str(data[k].dtypes):
                gg=data[k].describe().reset_index(drop=False)
                for l in gg["index"]:
                    data_number.loc[k,l]=gg[gg["index"]==l][k].iloc[0]
            elif "object" in str(data[k].dtypes):
                gg1=data[k].value_counts().reset_index(drop=False).rename(columns={"index":"value",k:"counts"})
                gg1["varname"]=k
                data_char=data_char.append(gg1)
    else:
        data_number=pd.DataFrame()
        data_char=pd.DataFrame()
        for k in tqdm(data.drop([id_name,y_name],axis=1).columns):
            if "int" in str(data[k].dtypes ) or "float" in str(data[k].dtypes):
                gg=data[k].describe().reset_index(drop=False)
                for l in gg["index"]:
                    data_number.loc[k,l]=gg[gg["index"]==l][k].iloc[0]
                data_number.loc[k,"缺失比例"]=1-gg[gg["index"]=="count"][k].iloc[0]/len(data)
            elif "object" in str(data[k].dtypes):
                gg1=data[k].value_counts().reset_index(drop=False).rename(columns={"index":"value",k:"counts"})
                gg1["varname"]=k
                gg1["缺失比例"]=1-data[k].count()/len(data)
                data_char=data_char.append(gg1)    
    return data_number,data_char
if __name__ == '__main__':
    train_data = pd.read_csv(r"F:\论文-风控\train.csv")  #读取训练集
    data1,data2=valueCounts(train_data,"id",y_name="isDefault")

