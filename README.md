## AugustBetty能做什么
  这个包是由微信公众号"屁屁和铭仔的数据之路"博主铭仔和屁屁一起开发的，目的是为了将逻辑回归模型开发流程规范化，
## 安装
    pip install AugustBetty
## 导入
    from AugustBetty.AugustBetty import ModelMain
## 一句话建模
     data = pd.read_csv(r"F:\论文-风控\train.csv")  #读取训练集
     m=MakeModel(data,"id","isDefault",0.02,5,512254,drop_value=["issueDate","earliesCreditLine"])
     m.model_main()

## v1.0.0的功能如下：
  #### 数值变量的最优分组
  输入指定的参数即可进行最优分组
  分组结果包含以下四项：
       1. 已经包含最优分组后的变量的数据集（在原始数据集上增加分组后的变量）
	   2. 最优分组后的变量名以及没有进行最优分组的变量（进行是组数较少，不进行最优分组）
	   3. 参与最优分组的变量的映射口径以及转换之后的变量名
	   4. 计算输入变量中转换了的变量的iv以及woe.
  #### 字符变量的最优分组
       1. 已经包含最优分组后的变量的数据集（在原始数据集上增加分组后的变量）
	   2. 最优分组后的变量名以及没有进行最优分组的变量（进行是组数较少，不进行最优分组）
	   3. 参与最优分组的变量的映射口径以及转换之后的变量名
	   4. 计算输入变量中转换了的变量的iv以及woe.
  #### 逻辑回归模型的开发
1.   训练集测试集的自动分区
2.   自动识别数据集中的字符以及数值变量进行最优分组
3.   自动转woe
4.   自动根据iv设定阈值，筛选出高于iv值的变量进入模型中拟合
5.   测试集根据分组映射woe,计算测试集的ks&auc,模型最终结果是模型的summary信息，以及测试集和训练集的ks&auc结果。

## V1.1.0的功能规划如下：
拆分出细致的单独功能：
1、单独的函数转原始数据
2、增加验证集（在原始数据集上使用时间参数，划分验证集，跑模型得出评估指标）
3、在输入的路径下，输出模型文件


## V1.2.0的功能规划如下：
开发自动部署模型的函数，输入模型位置，以及数据转换文件，输出模型api,以支持部分公司微服务的需求


如有报错，或者其他需求，请发邮件到：watchmans@qq.com 希望能不断的完善这个包，这个包现阶段功能相对单一。
欢迎大家关注公众号：屁屁和铭仔的数据之路


### 数值变量分组调用
 执行命令：`gg=num_group_chi_main(train_data,id_name,y,initial_group_num,group_num)`

 **train_data**:原始数据集
 
 **id_name**:主键名字，例如"idno"
 
 **y**:y变量名,例如"isbad"
 
 **initial_group_num**:粗分组组数，这里的逻辑是将数值变量先粗分组再细分组的方式进行最优分组，一般设定为20-50，看数据颗粒度
 
 **group_num**：最优分组组数，一般设定为4-6组
```python
 dict_keys(['data', 'var_dict_total', 'var_bin_list', 'iv_dict'])
 ```
data:增加新变量之后的数据集。原始变量没有改动。只对数值变量做分组。
 ```python
{'data':             id  loanAmnt  term  ...  n13_group_Bin  n14_group n14_group_Bin
 531009  531009   18000.0     5  ...          Bin 0          1         Bin 0
 370965  370965    3000.0     5  ...          Bin 0          1         Bin 0
 212592  212592   23600.0     3  ...          Bin 0          1         Bin 0
 610540  610540   20000.0     3  ...          Bin 0          1         Bin 0
 400590  400590   24250.0     3  ...          Bin 0          1         Bin 0
       ...       ...   ...  ...            ...        ...           ...
 603509  603509    5600.0     3  ...          Bin 0          7         Bin 4
 157628  157628    7200.0     3  ...          Bin 0          7         Bin 4
 147487  147487   10000.0     3  ...          Bin 0          7         Bin 4
 719781  719781   16000.0     3  ...          Bin 0          7         Bin 4
 735444  735444   20000.0     3  ...          Bin 0          7         Bin 4
 
 [800000 rows x 119 columns],
 ```
var_dict_total:保存的是变量的分组口径
 ```python
 'var_dict_total': {'loanAmnt': {'bin': {'Bin 0': [-9999.0, 8000.0],
    'Bin 1': [8000.0, 10000.0],
    'Bin 2': [10000.0, 15000.0],
    'Bin 3': [15000.0, 28000.0],
    'Bin 4': [28000.0, 10000000000]},
   'value_name': 'loanAmnt_group_Bin'},
  'interestRate': {'bin': {'Bin 0': [-9999.0, 7.97],
    'Bin 1': [7.97, 12.29],
    'Bin 2': [12.29, 15.99],
    'Bin 3': [15.99, 22.15],
    'Bin 4': [22.15, 10000000000]},
   'value_name': 'interestRate_group_Bin'},
  'installment': {'bin': {'Bin 0': [-9999.0, 248.45],
    'Bin 1': [248.45, 324.3],
    'Bin 2': [324.3, 10000000000]},
   'value_name': 'installment_group_Bin'},
  'employmentTitle': {'bin': {'Bin 0': [-9999.0, 54.0],
    'Bin 1': [54.0, 169034.0],
    'Bin 2': [169034.0, 10000000000]},
   'value_name': 'employmentTitle_group_Bin'},
  'homeOwnership': {'bin': {'Bin 0': [-9999.0, 1], 'Bin 1': [1, 10000000000]},
   'value_name': 'homeOwnership_group_Bin'},
  'annualIncome': {'bin': {'Bin 0': [-9999.0, 45600.0],
    'Bin 1': [45600.0, 60000.0],
    'Bin 2': [60000.0, 85000.0],
    'Bin 3': [85000.0, 125000.0],
    'Bin 4': [125000.0, 10000000000]},
   'value_name': 'annualIncome_group_Bin'},
  'purpose': {'bin': {'Bin 0': [-9999.0, 2], 'Bin 1': [2, 10000000000]},
   'value_name': 'purpose_group_Bin'},
  'postCode': {'bin': {'Bin 0': [-9999.0, 19.0],
    'Bin 1': [19.0, 262.0],
    'Bin 2': [262.0, 10000000000]},
   'value_name': 'postCode_group_Bin'},
  'regionCode': {'bin': {'Bin 0': [-9999.0, 13], 'Bin 1': [13, 10000000000]},
   'value_name': 'regionCode_group_Bin'},
  'dti': {'bin': {'Bin 0': [-9999.0, 14.18],
    'Bin 1': [14.18, 21.25],
    'Bin 2': [21.25, 25.69],
    'Bin 3': [25.69, 29.78],
    'Bin 4': [29.78, 10000000000]},
   'value_name': 'dti_group_Bin'},
  'delinquency_2years': {'bin': {'Bin 0': [-9999.0, 1.0],
    'Bin 1': [1.0, 2.0],
    'Bin 2': [2.0, 10000000000]},
   'value_name': 'delinquency_2years_group_Bin'},
  'ficoRangeLow': {'bin': {'Bin 0': [-9999.0, 670.0],
    'Bin 1': [670.0, 690.0],
    'Bin 2': [690.0, 720.0],
    'Bin 3': [720.0, 760.0],
    'Bin 4': [760.0, 10000000000]},
   'value_name': 'ficoRangeLow_group_Bin'},
  'ficoRangeHigh': {'bin': {'Bin 0': [-9999.0, 674.0],
    'Bin 1': [674.0, 694.0],
    'Bin 2': [694.0, 724.0],
    'Bin 3': [724.0, 764.0],
    'Bin 4': [764.0, 10000000000]},
   'value_name': 'ficoRangeHigh_group_Bin'},
  'openAcc': {'bin': {'Bin 0': [-9999.0, 7.0],
    'Bin 1': [7.0, 10.0],
    'Bin 2': [10.0, 17.0],
    'Bin 3': [17.0, 22.0],
    'Bin 4': [22.0, 10000000000]},
   'value_name': 'openAcc_group_Bin'},
  'pubRec': {'bin': {'Bin 0': [-9999.0, 1.0], 'Bin 1': [1.0, 10000000000]},
   'value_name': 'pubRec_group_Bin'},
  'pubRecBankruptcies': {'bin': {'Bin 0': [-9999.0, 0.0],
    'Bin 1': [0.0, 1.0],
    'Bin 2': [1.0, 10000000000]},
   'value_name': 'pubRecBankruptcies_group_Bin'},
  'revolBal': {'bin': {'Bin 0': [-9999.0, 22682.0],
    'Bin 1': [22682.0, 43290.0],
    'Bin 2': [43290.0, 10000000000]},
   'value_name': 'revolBal_group_Bin'},
  'revolUtil': {'bin': {'Bin 0': [-9999.0, 18.1],
    'Bin 1': [18.1, 33.4],
    'Bin 2': [33.4, 52.1],
    'Bin 3': [52.1, 74.9],
    'Bin 4': [74.9, 10000000000]},
   'value_name': 'revolUtil_group_Bin'},
  'totalAcc': {'bin': {'Bin 0': [-9999.0, 16.0],
    'Bin 1': [16.0, 22.0],
    'Bin 2': [22.0, 10000000000]},
   'value_name': 'totalAcc_group_Bin'},
  'title': {'bin': {'Bin 0': [-9999.0, 1.0],
    'Bin 1': [1.0, 10.0],
    'Bin 2': [10.0, 10000000000]},
   'value_name': 'title_group_Bin'},
  'n0': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n0_group_Bin'},
  'n1': {'bin': {'Bin 0': [-9999.0, 2.0],
    'Bin 1': [2.0, 3.0],
    'Bin 2': [3.0, 4.0],
    'Bin 3': [4.0, 6.0],
    'Bin 4': [6.0, 10000000000]},
   'value_name': 'n1_group_Bin'},
  'n2': {'bin': {'Bin 0': [-9999.0, 2.0],
    'Bin 1': [2.0, 4.0],
    'Bin 2': [4.0, 7.0],
    'Bin 3': [7.0, 10.0],
    'Bin 4': [10.0, 10000000000]},
   'value_name': 'n2_group_Bin'},
  'n3': {'bin': {'Bin 0': [-9999.0, 2.0],
    'Bin 1': [2.0, 4.0],
    'Bin 2': [4.0, 7.0],
    'Bin 3': [7.0, 10.0],
    'Bin 4': [10.0, 10000000000]},
   'value_name': 'n3_group_Bin'},
  'n4': {'bin': {'Bin 0': [-9999.0, 0.0],
    'Bin 1': [0.0, 4.0],
    'Bin 2': [4.0, 6.0],
    'Bin 3': [6.0, 10.0],
    'Bin 4': [10.0, 10000000000]},
   'value_name': 'n4_group_Bin'},
  'n5': {'bin': {'Bin 0': [-9999.0, 4.0], 'Bin 1': [4.0, 10000000000]},
   'value_name': 'n5_group_Bin'},
  'n6': {'bin': {'Bin 0': [-9999.0, 1.0],
    'Bin 1': [1.0, 23.0],
    'Bin 2': [23.0, 10000000000]},
   'value_name': 'n6_group_Bin'},
  'n7': {'bin': {'Bin 0': [-9999.0, 4.0],
    'Bin 1': [4.0, 7.0],
    'Bin 2': [7.0, 10.0],
    'Bin 3': [10.0, 14.0],
    'Bin 4': [14.0, 10000000000]},
   'value_name': 'n7_group_Bin'},
  'n8': {'bin': {'Bin 0': [-9999.0, 5.0], 'Bin 1': [5.0, 10000000000]},
   'value_name': 'n8_group_Bin'},
  'n9': {'bin': {'Bin 0': [-9999.0, 3.0],
    'Bin 1': [3.0, 5.0],
    'Bin 2': [5.0, 7.0],
    'Bin 3': [7.0, 10.0],
    'Bin 4': [10.0, 10000000000]},
   'value_name': 'n9_group_Bin'},
  'n10': {'bin': {'Bin 0': [-9999.0, 3.0],
    'Bin 1': [3.0, 8.0],
    'Bin 2': [8.0, 10.0],
    'Bin 3': [10.0, 22.0],
    'Bin 4': [22.0, 10000000000]},
   'value_name': 'n10_group_Bin'},
  'n11': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n11_group_Bin'},
  'n12': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n12_group_Bin'},
  'n13': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n13_group_Bin'},
  'n14': {'bin': {'Bin 0': [-9999.0, 0.0],
    'Bin 1': [0.0, 1.0],
    'Bin 2': [1.0, 2.0],
    'Bin 3': [2.0, 3.0],
    'Bin 4': [3.0, 10000000000]},
   'value_name': 'n14_group_Bin'}},
 ```
var_bin_list：保存可用于如模型拟合的变量（逻辑回归本质上入模的变量都应该是数值变量，这里的list保存的时候分组好的变量以及原本的分组组已经满足组数（一般设定5组）的变量list,输出这个的原因是后续好循环转换woe,无需手工把变量敲出来。）
  ```python
'var_bin_list': ['loanAmnt_group_Bin',
  'interestRate_group_Bin',
  'installment_group_Bin',
  'employmentTitle_group_Bin',
  'homeOwnership_group_Bin',
  'annualIncome_group_Bin',
  'purpose_group_Bin',
  'postCode_group_Bin',
  'regionCode_group_Bin',
  'dti_group_Bin',
  'delinquency_2years_group_Bin',
  'ficoRangeLow_group_Bin',
  'ficoRangeHigh_group_Bin',
  'openAcc_group_Bin',
  'pubRec_group_Bin',
  'pubRecBankruptcies_group_Bin',
  'revolBal_group_Bin',
  'revolUtil_group_Bin',
  'totalAcc_group_Bin',
  'title_group_Bin',
  'n0_group_Bin',
  'n1_group_Bin',
  'n2_group_Bin',
  'n3_group_Bin',
  'n4_group_Bin',
  'n5_group_Bin',
  'n6_group_Bin',
  'n7_group_Bin',
  'n8_group_Bin',
  'n9_group_Bin',
  'n10_group_Bin',
  'n11_group_Bin',
  'n12_group_Bin',
  'n13_group_Bin',
  'n14_group_Bin',
  'term',
  'verificationStatus',
  'initialListStatus',
  'applicationType',
  'policyCode'],
  ```
iv_dict：保存是的输入的数据集中可最优分组的变量，转换后的iv,woe,以及已经是足够组数的变量的iv,woe
  ```python
'iv_dict': {'term': {'iv': 0.17263507884380802,
   'woe': {3: -0.2686182828015269, 5: 0.6520081414240493}},
  'verificationStatus': {'iv': 0.05451891242377009,
   'woe': {0: -0.3672196958578996,
    1: 0.06087198832351855,
    2: 0.22488431630897235}},
  'initialListStatus': {'iv': 0.0003419546491836722,
   'woe': {0: 0.015549994613464051, 1: -0.021991286981396775}},
  'applicationType': {'iv': 0.001906255272074922,
   'woe': {0: -0.006394473785834787, 1: 0.2981570883254602}},
  'policyCode': {'iv': 0.0, 'woe': {1.0: 0.0}},
  'loanAmnt_group_Bin': {'iv': 0.03450643699626254,
   'woe': {'Bin 0': -0.26698353031170635,
    'Bin 1': -0.10585087483552284,
    'Bin 2': 0.05207841143835551,
    'Bin 3': 0.15704067216706727,
    'Bin 4': 0.2587985257714207}},
  'interestRate_group_Bin': {'iv': 0.432844128756394,
   'woe': {'Bin 0': -1.4139767754271722,
    'Bin 1': -0.47444483191986947,
    'Bin 2': 0.12600617373518758,
    'Bin 3': 0.6329403904111698,
    'Bin 4': 1.1364715170302837}},
  'installment_group_Bin': {'iv': 0.028446931068475675,
   'woe': {'Bin 0': -0.2965362458671355,
    'Bin 1': -0.027032933155407588,
    'Bin 2': 0.11545123400239903}},
  'employmentTitle_group_Bin': {'iv': 0.01199262426711711,
   'woe': {'Bin 0': 0.1974127496581969,
    'Bin 1': 0.015794613325623683,
    'Bin 2': -0.19061653548410487}},
  'homeOwnership_group_Bin': {'iv': 0.0003224362597742428,
   'woe': {'Bin 0': -0.006310542058851011, 'Bin 1': 0.051096232036222884}},
  'annualIncome_group_Bin': {'iv': 0.028755712221678248,
   'woe': {'Bin 0': 0.19386504291971357,
    'Bin 1': 0.09133649436575257,
    'Bin 2': -0.026827665423123023,
    'Bin 3': -0.19469768671127013,
    'Bin 4': -0.339204930328438}},
  'purpose_group_Bin': {'iv': 0.007295208182839682,
   'woe': {'Bin 0': 0.060104555523002035, 'Bin 1': -0.12144911490483187}},
  'postCode_group_Bin': {'iv': 0.0008119510164983021,
   'woe': {'Bin 0': -0.10613844771754728,
    'Bin 1': -0.006970955095481508,
    'Bin 2': 0.02290230125349964}},
  'regionCode_group_Bin': {'iv': 1.3326503587742891e-05,
   'woe': {'Bin 0': 0.0037306864251708577, 'Bin 1': -0.003572135757512189}},
  'dti_group_Bin': {'iv': 0.0691444505823585,
   'woe': {'Bin 0': -0.29639426975460115,
    'Bin 1': -0.05303861299797819,
    'Bin 2': 0.14116987973826364,
    'Bin 3': 0.2922661709950033,
    'Bin 4': 0.49293116504622203}},
  'delinquency_2years_group_Bin': {'iv': 0.002139337579274896,
   'woe': {'Bin 0': -0.012315786189818509,
    'Bin 1': 0.14386566036838583,
    'Bin 2': 0.20305465903034065}},
  'ficoRangeLow_group_Bin': {'iv': 0.11849558508830776,
   'woe': {'Bin 0': 0.33499130235102315,
    'Bin 1': 0.14036402817126167,
    'Bin 2': -0.16102580213328785,
    'Bin 3': -0.582451594346089,
    'Bin 4': -1.0396374414819025}},
  'ficoRangeHigh_group_Bin': {'iv': 0.11849558508830776,
   'woe': {'Bin 0': 0.33499130235102315,
    'Bin 1': 0.14036402817126167,
    'Bin 2': -0.16102580213328785,
    'Bin 3': -0.582451594346089,
    'Bin 4': -1.0396374414819025}},
  'openAcc_group_Bin': {'iv': 0.004573486625874178,
   'woe': {'Bin 0': -0.0890448653651065,
    'Bin 1': -0.030985128792218982,
    'Bin 2': 0.03156043510552112,
    'Bin 3': 0.08247580996942927,
    'Bin 4': 0.18573680489483868}},
  'pubRec_group_Bin': {'iv': 0.001300898810070593,
   'woe': {'Bin 0': -0.006265183515565402, 'Bin 1': 0.20766187334039157}},
  'pubRecBankruptcies_group_Bin': {'iv': 0.004275157313198444,
   'woe': {'Bin 0': -0.025198889500429856,
    'Bin 1': 0.16242870100858983,
    'Bin 2': 0.2476354559457032}},
  'revolBal_group_Bin': {'iv': 0.003766992022044162,
   'woe': {'Bin 0': 0.02404226891591574,
    'Bin 1': -0.05292092268354001,
    'Bin 2': -0.24978853781535482}},
  'revolUtil_group_Bin': {'iv': 0.02397767300459437,
   'woe': {'Bin 0': -0.358979455056802,
    'Bin 1': -0.17312304317097293,
    'Bin 2': -0.021333642551765307,
    'Bin 3': 0.09298419081740512,
    'Bin 4': 0.1599708320355574}},
  'totalAcc_group_Bin': {'iv': 0.001774565948034374,
   'woe': {'Bin 0': 0.0654454812926487,
    'Bin 1': 0.0046315344580754,
    'Bin 2': -0.03502643818846728}},
  'title_group_Bin': {'iv': 0.02138620287983934,
   'woe': {'Bin 0': 0.12076691332729704,
    'Bin 1': -0.06234571640663273,
    'Bin 2': -0.303010040073495}},
  'n0_group_Bin': {'iv': 0.0036482088372045843,
   'woe': {'Bin 0': -0.03328497196626471, 'Bin 1': 0.10963861162794214}},
  'n1_group_Bin': {'iv': 0.012776531508686802,
   'woe': {'Bin 0': -0.1139367854446227,
    'Bin 1': -0.03145428250504604,
    'Bin 2': 0.029134607317330145,
    'Bin 3': 0.10471618734510002,
    'Bin 4': 0.24003074689416587}},
  'n2_group_Bin': {'iv': 0.03305847723307784,
   'woe': {'Bin 0': -0.2599793507860163,
    'Bin 1': -0.1159979609788405,
    'Bin 2': 0.03678331487568212,
    'Bin 3': 0.19935833147539536,
    'Bin 4': 0.3818852049766906}},
  'n3_group_Bin': {'iv': 0.03305847723307784,
   'woe': {'Bin 0': -0.2599793507860163,
    'Bin 1': -0.1159979609788405,
    'Bin 2': 0.03678331487568212,
    'Bin 3': 0.19935833147539536,
    'Bin 4': 0.3818852049766906}},
  'n4_group_Bin': {'iv': 0.005517283907591153,
   'woe': {'Bin 0': -0.2889421330249432,
    'Bin 1': -0.013202349900056868,
    'Bin 2': 0.019088384219396928,
    'Bin 3': 0.06247232677779756,
    'Bin 4': 0.13402280957203358}},
  'n5_group_Bin': {'iv': 0.00020945793402011507,
   'woe': {'Bin 0': 0.023494043351489774, 'Bin 1': -0.008915518979970272}},
  'n6_group_Bin': {'iv': 0.0014692680358088994,
   'woe': {'Bin 0': -0.0765558185257892,
    'Bin 1': 0.0053247020248672955,
    'Bin 2': 0.1210950403409298}},
  'n7_group_Bin': {'iv': 0.009675687051079466,
   'woe': {'Bin 0': -0.14429755533030605,
    'Bin 1': -0.027727108030571924,
    'Bin 2': 0.041131450162872625,
    'Bin 3': 0.08806082871714123,
    'Bin 4': 0.18795745190250174}},
  'n8_group_Bin': {'iv': 0.0007184262698178514,
   'woe': {'Bin 0': -0.06987676998104353, 'Bin 1': 0.010281947543667881}},
  'n9_group_Bin': {'iv': 0.031249289588835223,
   'woe': {'Bin 0': -0.2096436342241601,
    'Bin 1': -0.0429081760617028,
    'Bin 2': 0.07105221873661946,
    'Bin 3': 0.1970894208174422,
    'Bin 4': 0.3774005738856138}},
  'n10_group_Bin': {'iv': 0.008444830436101544,
   'woe': {'Bin 0': -0.293226157535438,
    'Bin 1': -0.06274924779747353,
    'Bin 2': -0.00328536625477653,
    'Bin 3': 0.05386221291998551,
    'Bin 4': 0.19061556079370015}},
  'n11_group_Bin': {'iv': 7.421631321053656e-08,
   'woe': {'Bin 0': 7.237673677632179e-06, 'Bin 1': -0.010254233468312956}},
  'n12_group_Bin': {'iv': 7.759942856943026e-05,
   'woe': {'Bin 0': -0.0004960486167429998, 'Bin 1': 0.15643612383430344}},
  'n13_group_Bin': {'iv': 0.0015580211455861362,
   'woe': {'Bin 0': -0.009717341239876488, 'Bin 1': 0.1603549088517746}},
  'n14_group_Bin': {'iv': 0.048440899304046665,
   'woe': {'Bin 0': -0.3045306871267572,
    'Bin 1': -0.14851272009664643,
    'Bin 2': 0.0023652073777263066,
    'Bin 3': 0.14618635044605022,
    'Bin 4': 0.3310966221715067}}}}
  ```



### 字符变量分组调用

char_group_chi_main(data_total,id_name,y,group_num,drop_value=[])

 **data_total**:原始数据集
 
 **id_name**:主键名字，例如"idno"
 
 **y**:y变量名,例如"isbad"
  
 **group_num**：最优分组组数，一般设定为4-6组
 
 **drop_value**:不参与最优分组的变量，例如字符型的日期这一类变量。
 
 
 **例如以下填写参数**：
 ```python
 char_group_chi_main(train_data,"id","isDefault",5,drop_value=["issueDate","earliesCreditLine"])
 ```
**返回值**：与数值的变量的一样，不再赘述：
 ```python
{'data':             id  loanAmnt  ...  employmentLength_group  employmentLength_group_Bin
 0            0   35000.0  ...                       8                       Bin 2
 1            1   18000.0  ...                       4                       Bin 1
 2            2   12000.0  ...                       5                       Bin 2
 3            3   11000.0  ...                       1                       Bin 0
 4            4    3000.0  ...                      12                       Bin 4
       ...       ...  ...                     ...                         ...
 799995  799995   25000.0  ...                       3                       Bin 1
 799996  799996   17000.0  ...                       1                       Bin 0
 799997  799997    6000.0  ...                       1                       Bin 0
 799998  799998   19200.0  ...                       1                       Bin 0
 799999  799999    9000.0  ...                       4                       Bin 1
 
 [800000 rows x 54 columns],
 'var_dict_total': {'grade': {'bin': {'F': 'Bin 4',
    'G': 'Bin 4',
    'D': 'Bin 3',
    'E': 'Bin 4',
    'C': 'Bin 2',
    'A': 'Bin 0',
    'B': 'Bin 1'},
   'var_name': 'grade_group_Bin'},
  'subGrade': {'bin': {'G1': 'Bin 4',
    'G2': 'Bin 4',
    'G3': 'Bin 4',
    'F4': 'Bin 4',
    'F3': 'Bin 4',
    'G4': 'Bin 4',
    'E3': 'Bin 4',
    'F5': 'Bin 4',
    'E4': 'Bin 4',
    'D3': 'Bin 3',
    'D2': 'Bin 3',
    'F1': 'Bin 4',
    'A2': 'Bin 0',
    'F2': 'Bin 4',
    'A5': 'Bin 0',
    'A3': 'Bin 0',
    'B5': 'Bin 1',
    'B4': 'Bin 1',
    'E5': 'Bin 4',
    'G5': 'Bin 4',
    'D1': 'Bin 3',
    'E1': 'Bin 4',
    'C1': 'Bin 2',
    'D4': 'Bin 3',
    'B1': 'Bin 1',
    'D5': 'Bin 3',
    'C2': 'Bin 2',
    'B2': 'Bin 1',
    'C5': 'Bin 3',
    'A1': 'Bin 0',
    'E2': 'Bin 4',
    'C4': 'Bin 3',
    'C3': 'Bin 2',
    'A4': 'Bin 0',
    'B3': 'Bin 1'},
   'var_name': 'subGrade_group_Bin'},
  'employmentLength': {'bin': {'6 years': 'Bin 1',
    '3 years': 'Bin 2',
    '9 years': 'Bin 2',
    '4 years': 'Bin 2',
    '< 1 year': 'Bin 3',
    '8 years': 'Bin 2',
    '1 year': 'Bin 3',
    'null': 'Bin 4',
    '2 years': 'Bin 2',
    '7 years': 'Bin 1',
    '10+ years': 'Bin 0',
    '5 years': 'Bin 1'},
   'var_name': 'employmentLength_group_Bin'}},
 'var_bin_list': ['grade_group_Bin',
  'subGrade_group_Bin',
  'employmentLength_group_Bin'],
 'iv_dict': {'grade_group_Bin': {'iv': 0.46008926359723423,
   'woe': {'Bin 0': -1.355565869742944,
    'Bin 1': -0.4854115082757125,
    'Bin 2': 0.15269375877204816,
    'Bin 3': 0.560321181232565,
    'Bin 4': 1.0184060522289387}},
  'subGrade_group_Bin': {'iv': 0.46133308496995795,
   'woe': {'Bin 0': -1.355565869742944,
    'Bin 1': -0.4854115082757125,
    'Bin 2': 0.04570655110165618,
    'Bin 3': 0.46384538567636546,
    'Bin 4': 1.0184060522289387}},
  'employmentLength_group_Bin': {'iv': 0.012227612557761728,
   'woe': {'Bin 0': -0.08002342137428835,
    'Bin 1': -0.03072040532440133,
    'Bin 2': -0.00026093587689965793,
    'Bin 3': 0.03412595689559949,
    'Bin 4': 0.38856251491409827}}}}
```
 
 #### 一个语句建模
 刚才的最优分组都是可以单独使用的，一个语句建模，即只需要执行一个语句，则能将原始的数据集从分组、转换woe、筛选变量，拟合模型、最终得到测试集以及训练集的结果。
 
 定义模型参数：
 ```python
     m=MakeModel(data,"id","isDefault",0.02,5,512254,drop_value=["issueDate","earliesCreditLine"])
     def __init__(self,data,id_name,y_name,iv_num,group_num,random_state,drop_value=[]):

```
**data**:原始数据集，最好是你处理过的数据集，具体的衍生变量你自己做啦~~~

**id_name**：主键的变量名 例如 "idno"

**iv_num**：设定iv大于多少就可以进入模型拟合 建议是0.01-0.05之间

**y_name**:y变量名 例如"isbad"

**group_num**:最优分组最终组数，最好的4-6之间

**random_state**：划分数据集的随机码

**drop_value**：不参与建模的变量，你可以自己在原始数据集里面删掉，也可以在这写上。

###**执行语句**
 ```python
    m=MakeModel(data,"id","isDefault",0.02,5,512254,drop_value=["issueDate","earliesCreditLine"])
    m.model_main()
 ```
最终m包含以下信息：
 **bin_woe** 这是类中定义的一个用户转换woe的函数，不管！！
 
 **char_object**：这个是最终字符变量最优分组输出的信息，上面贴过了，我再贴一次，因为我知道你懒得翻回去
  ```python
{'data':             id  loanAmnt  ...  employmentLength_group  employmentLength_group_Bin
 742562  742562   10000.0  ...                       1                       Bin 0
 762848  762848   14950.0  ...                       6                       Bin 2
 644240  644240   20000.0  ...                      12                       Bin 4
 451072  451072   12000.0  ...                       9                       Bin 3
 93411    93411    2500.0  ...                      11                       Bin 3
       ...       ...  ...                     ...                         ...
 245245  245245    4450.0  ...                       9                       Bin 3
 30730    30730    5700.0  ...                       1                       Bin 0
 476922  476922    7500.0  ...                       2                       Bin 1
 476992  476992   15000.0  ...                      10                       Bin 3
 39730    39730   40000.0  ...                       2                       Bin 1
 
 [560000 rows x 56 columns],
 'var_dict_total': {'grade': {'bin': {'G': 'Bin 4',
    'E': 'Bin 4',
    'F': 'Bin 4',
    'D': 'Bin 3',
    'C': 'Bin 2',
    'B': 'Bin 1',
    'A': 'Bin 0'},
   'var_name': 'grade_group_Bin'},
  'subGrade': {'bin': {'G5': 'Bin 4',
    'G4': 'Bin 4',
    'F2': 'Bin 4',
    'G3': 'Bin 4',
    'G2': 'Bin 4',
    'F5': 'Bin 4',
    'F4': 'Bin 4',
    'E3': 'Bin 4',
    'G1': 'Bin 4',
    'E4': 'Bin 4',
    'A2': 'Bin 0',
    'A1': 'Bin 0',
    'A3': 'Bin 0',
    'E1': 'Bin 3',
    'D1': 'Bin 3',
    'C4': 'Bin 3',
    'D2': 'Bin 3',
    'E5': 'Bin 4',
    'C3': 'Bin 3',
    'D5': 'Bin 3',
    'E2': 'Bin 4',
    'B4': 'Bin 2',
    'F1': 'Bin 4',
    'C5': 'Bin 3',
    'B1': 'Bin 1',
    'C1': 'Bin 2',
    'D4': 'Bin 3',
    'A4': 'Bin 0',
    'B2': 'Bin 1',
    'F3': 'Bin 4',
    'D3': 'Bin 3',
    'C2': 'Bin 2',
    'B5': 'Bin 2',
    'B3': 'Bin 1',
    'A5': 'Bin 1'},
   'var_name': 'subGrade_group_Bin'},
  'employmentLength': {'bin': {'9 years': 'Bin 2',
    '4 years': 'Bin 2',
    '8 years': 'Bin 2',
    '7 years': 'Bin 1',
    '5 years': 'Bin 1',
    'null': 'Bin 4',
    '1 year': 'Bin 3',
    '2 years': 'Bin 2',
    '3 years': 'Bin 3',
    '10+ years': 'Bin 0',
    '< 1 year': 'Bin 3',
    '6 years': 'Bin 1'},
   'var_name': 'employmentLength_group_Bin'}},
 'var_bin_list': ['grade_group_Bin',
  'subGrade_group_Bin',
  'employmentLength_group_Bin'],
 'iv_dict': {'grade_group_Bin': {'iv': 0.46497204120261115,
   'woe': {'Bin 0': -1.366114993583636,
    'Bin 1': -0.4872500245227714,
    'Bin 2': 0.15134427847202556,
    'Bin 3': 0.5673843591805174,
    'Bin 4': 1.0199366732850919}},
  'subGrade_group_Bin': {'iv': 0.46933032966767474,
   'woe': {'Bin 0': -1.5499290796800491,
    'Bin 1': -0.7155043413125765,
    'Bin 2': -0.14179350218267067,
    'Bin 3': 0.43440570785349064,
    'Bin 4': 1.0693024724239095}},
  'employmentLength_group_Bin': {'iv': 0.011729759860744725,
   'woe': {'Bin 0': -0.07860890366978489,
    'Bin 1': -0.030637664745026258,
    'Bin 2': -0.005312943616562383,
    'Bin 3': 0.02768067382373055,
    'Bin 4': 0.3796614757205021}}}}
   ```

 **corr_x**:定义的一个计算相关系数矩阵的函数，不管！！！
 
 **deal_raw_data**：处理映射的函数，不管！！
 
 **drop_value**： 你输入的删掉的变量的参数
 
 **feature_list**：进入模型拟合的变量列表，如果你对选出来的变量不满意，可以自己选变量，这样子跑就可以了
   ```python
    m=MakeModel(data,"id","isDefault",0.02,5,512254,drop_value=["issueDate","earliesCreditLine"])
    m.model_main(feature_list=["aa","bb","dd","cc"])
  ```
 **group_num**:你输入的分组数
 
 **id_name**：你输入的主键的变量名
 
  **iv_num**：你输入的iv阈值
  
 **ks_calc_auc**：一个计算ks的函数，不用管。你想用，我之后给你放出来，这个版本不放
 
 **model_main**：建模主函数，不用管
 
 **num_object**：这个是最终数值变量最优分组输出的信息，上面贴过了，我再贴一次，因为我知道你懒得翻回去
 ```python 
{'data':             id  loanAmnt  term  ...  n13_group_Bin  n14_group n14_group_Bin
 384065  384065    7500.0     3  ...          Bin 0          1         Bin 0
 416852  416852   35000.0     5  ...          Bin 0          1         Bin 0
 794591  794591   30000.0     5  ...          Bin 0          1         Bin 0
 686008  686008    5000.0     3  ...          Bin 0          1         Bin 0
 449392  449392   12000.0     3  ...          Bin 0          1         Bin 0
       ...       ...   ...  ...            ...        ...           ...
 764543  764543   12000.0     3  ...          Bin 0          7         Bin 4
 230131  230131    5550.0     3  ...          Bin 1          7         Bin 4
 289950  289950    2200.0     3  ...          Bin 0          7         Bin 4
 95878    95878    5500.0     3  ...          Bin 1          7         Bin 4
 157628  157628    7200.0     3  ...          Bin 0          7         Bin 4
 
 [560000 rows x 117 columns],
 'var_dict_total': {'loanAmnt': {'bin': {'Bin 0': [-9999.0, 8625.0],
    'Bin 1': [8625.0, 10000.0],
    'Bin 2': [10000.0, 15000.0],
    'Bin 3': [15000.0, 28000.0],
    'Bin 4': [28000.0, 10000000000]},
   'value_name': 'loanAmnt_group_Bin'},
  'interestRate': {'bin': {'Bin 0': [-9999.0, 7.97],
    'Bin 1': [7.97, 12.29],
    'Bin 2': [12.29, 15.99],
    'Bin 3': [15.99, 22.15],
    'Bin 4': [22.15, 10000000000]},
   'value_name': 'interestRate_group_Bin'},
  'installment': {'bin': {'Bin 0': [-9999.0, 248.25],
    'Bin 1': [248.25, 324.07],
    'Bin 2': [324.07, 10000000000]},
   'value_name': 'installment_group_Bin'},
  'employmentTitle': {'bin': {'Bin 0': [-9999.0, 54.0],
    'Bin 1': [54.0, 169060.0],
    'Bin 2': [169060.0, 10000000000]},
   'value_name': 'employmentTitle_group_Bin'},
  'homeOwnership': {'bin': {'Bin 0': [-9999.0, 1],
    'Bin 1': [1, 2],
    'Bin 2': [2, 10000000000]},
   'value_name': 'homeOwnership_group_Bin'},
  'annualIncome': {'bin': {'Bin 0': [-9999.0, 42000.0],
    'Bin 1': [42000.0, 60000.0],
    'Bin 2': [60000.0, 85000.0],
    'Bin 3': [85000.0, 125000.0],
    'Bin 4': [125000.0, 10000000000]},
   'value_name': 'annualIncome_group_Bin'},
  'purpose': {'bin': {'Bin 0': [-9999.0, 2], 'Bin 1': [2, 10000000000]},
   'value_name': 'purpose_group_Bin'},
  'postCode': {'bin': {'Bin 0': [-9999.0, 19.0],
    'Bin 1': [19.0, 300.0],
    'Bin 2': [300.0, 10000000000]},
   'value_name': 'postCode_group_Bin'},
  'regionCode': {'bin': {'Bin 0': [-9999.0, 21], 'Bin 1': [21, 10000000000]},
   'value_name': 'regionCode_group_Bin'},
  'dti': {'bin': {'Bin 0': [-9999.0, 15.31],
    'Bin 1': [15.31, 21.25],
    'Bin 2': [21.25, 25.69],
    'Bin 3': [25.69, 29.78],
    'Bin 4': [29.78, 10000000000]},
   'value_name': 'dti_group_Bin'},
  'delinquency_2years': {'bin': {'Bin 0': [-9999.0, 1.0],
    'Bin 1': [1.0, 2.0],
    'Bin 2': [2.0, 10000000000]},
   'value_name': 'delinquency_2years_group_Bin'},
  'ficoRangeLow': {'bin': {'Bin 0': [-9999.0, 680.0],
    'Bin 1': [680.0, 705.0],
    'Bin 2': [705.0, 720.0],
    'Bin 3': [720.0, 760.0],
    'Bin 4': [760.0, 10000000000]},
   'value_name': 'ficoRangeLow_group_Bin'},
  'ficoRangeHigh': {'bin': {'Bin 0': [-9999.0, 684.0],
    'Bin 1': [684.0, 709.0],
    'Bin 2': [709.0, 724.0],
    'Bin 3': [724.0, 764.0],
    'Bin 4': [764.0, 10000000000]},
   'value_name': 'ficoRangeHigh_group_Bin'},
  'openAcc': {'bin': {'Bin 0': [-9999.0, 8.0],
    'Bin 1': [8.0, 10.0],
    'Bin 2': [10.0, 19.0],
    'Bin 3': [19.0, 22.0],
    'Bin 4': [22.0, 10000000000]},
   'value_name': 'openAcc_group_Bin'},
  'pubRec': {'bin': {'Bin 0': [-9999.0, 1.0], 'Bin 1': [1.0, 10000000000]},
   'value_name': 'pubRec_group_Bin'},
  'pubRecBankruptcies': {'bin': {'Bin 0': [-9999.0, 0.0],
    'Bin 1': [0.0, 1.0],
    'Bin 2': [1.0, 10000000000]},
   'value_name': 'pubRecBankruptcies_group_Bin'},
  'revolBal': {'bin': {'Bin 0': [-9999.0, 22687.0],
    'Bin 1': [22687.0, 43329.0],
    'Bin 2': [43329.0, 10000000000]},
   'value_name': 'revolBal_group_Bin'},
  'revolUtil': {'bin': {'Bin 0': [-9999.0, 18.1],
    'Bin 1': [18.1, 37.4],
    'Bin 2': [37.4, 55.6],
    'Bin 3': [55.6, 74.9],
    'Bin 4': [74.9, 10000000000]},
   'value_name': 'revolUtil_group_Bin'},
  'totalAcc': {'bin': {'Bin 0': [-9999.0, 16.0],
    'Bin 1': [16.0, 22.0],
    'Bin 2': [22.0, 26.0],
    'Bin 3': [26.0, 10000000000]},
   'value_name': 'totalAcc_group_Bin'},
  'title': {'bin': {'Bin 0': [-9999.0, 1.0],
    'Bin 1': [1.0, 10.0],
    'Bin 2': [10.0, 10000000000]},
   'value_name': 'title_group_Bin'},
  'n0': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n0_group_Bin'},
  'n1': {'bin': {'Bin 0': [-9999.0, 2.0],
    'Bin 1': [2.0, 3.0],
    'Bin 2': [3.0, 4.0],
    'Bin 3': [4.0, 6.0],
    'Bin 4': [6.0, 10000000000]},
   'value_name': 'n1_group_Bin'},
  'n2': {'bin': {'Bin 0': [-9999.0, 2.0],
    'Bin 1': [2.0, 4.0],
    'Bin 2': [4.0, 6.0],
    'Bin 3': [6.0, 8.0],
    'Bin 4': [8.0, 10000000000]},
   'value_name': 'n2_group_Bin'},
  'n3': {'bin': {'Bin 0': [-9999.0, 2.0],
    'Bin 1': [2.0, 4.0],
    'Bin 2': [4.0, 6.0],
    'Bin 3': [6.0, 8.0],
    'Bin 4': [8.0, 10000000000]},
   'value_name': 'n3_group_Bin'},
  'n4': {'bin': {'Bin 0': [-9999.0, 0.0],
    'Bin 1': [0.0, 4.0],
    'Bin 2': [4.0, 6.0],
    'Bin 3': [6.0, 10000000000]},
   'value_name': 'n4_group_Bin'},
  'n5': {'bin': {'Bin 0': [-9999.0, 4.0], 'Bin 1': [4.0, 10000000000]},
   'value_name': 'n5_group_Bin'},
  'n6': {'bin': {'Bin 0': [-9999.0, 1.0],
    'Bin 1': [1.0, 23.0],
    'Bin 2': [23.0, 10000000000]},
   'value_name': 'n6_group_Bin'},
  'n7': {'bin': {'Bin 0': [-9999.0, 4.0],
    'Bin 1': [4.0, 7.0],
    'Bin 2': [7.0, 10.0],
    'Bin 3': [10.0, 14.0],
    'Bin 4': [14.0, 10000000000]},
   'value_name': 'n7_group_Bin'},
  'n8': {'bin': {'Bin 0': [-9999.0, 5.0], 'Bin 1': [5.0, 10000000000]},
   'value_name': 'n8_group_Bin'},
  'n9': {'bin': {'Bin 0': [-9999.0, 3.0],
    'Bin 1': [3.0, 5.0],
    'Bin 2': [5.0, 7.0],
    'Bin 3': [7.0, 10.0],
    'Bin 4': [10.0, 10000000000]},
   'value_name': 'n9_group_Bin'},
  'n10': {'bin': {'Bin 0': [-9999.0, 3.0],
    'Bin 1': [3.0, 8.0],
    'Bin 2': [8.0, 10.0],
    'Bin 3': [10.0, 19.0],
    'Bin 4': [19.0, 10000000000]},
   'value_name': 'n10_group_Bin'},
  'n12': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n12_group_Bin'},
  'n13': {'bin': {'Bin 0': [-9999.0, 0.0], 'Bin 1': [0.0, 10000000000]},
   'value_name': 'n13_group_Bin'},
  'n14': {'bin': {'Bin 0': [-9999.0, 0.0],
    'Bin 1': [0.0, 1.0],
    'Bin 2': [1.0, 2.0],
    'Bin 3': [2.0, 3.0],
    'Bin 4': [3.0, 10000000000]},
   'value_name': 'n14_group_Bin'}},
 'var_bin_list': ['loanAmnt_group_Bin',
  'interestRate_group_Bin',
  'installment_group_Bin',
  'employmentTitle_group_Bin',
  'homeOwnership_group_Bin',
  'annualIncome_group_Bin',
  'purpose_group_Bin',
  'postCode_group_Bin',
  'regionCode_group_Bin',
  'dti_group_Bin',
  'delinquency_2years_group_Bin',
  'ficoRangeLow_group_Bin',
  'ficoRangeHigh_group_Bin',
  'openAcc_group_Bin',
  'pubRec_group_Bin',
  'pubRecBankruptcies_group_Bin',
  'revolBal_group_Bin',
  'revolUtil_group_Bin',
  'totalAcc_group_Bin',
  'title_group_Bin',
  'n0_group_Bin',
  'n1_group_Bin',
  'n2_group_Bin',
  'n3_group_Bin',
  'n4_group_Bin',
  'n5_group_Bin',
  'n6_group_Bin',
  'n7_group_Bin',
  'n8_group_Bin',
  'n9_group_Bin',
  'n10_group_Bin',
  'n12_group_Bin',
  'n13_group_Bin',
  'n14_group_Bin',
  'term',
  'verificationStatus',
  'initialListStatus',
  'applicationType',
  'policyCode',
  'n11'],
 'iv_dict': {'term': {'iv': 0.17475272449362428,
   'woe': {3: -0.2703191679368651, 5: 0.6559694111951718}},
  'verificationStatus': {'iv': 0.053778743185793496,
   'woe': {0: -0.36484279075332327,
    1: 0.06141693489701235,
    2: 0.22266031298696004}},
  'initialListStatus': {'iv': 0.00038571827775202977,
   'woe': {0: 0.01649700561049484, 1: -0.023381860067453963}},
  'applicationType': {'iv': 0.001984746493023151,
   'woe': {0: -0.006543082114183424, 1: 0.3033852042307132}},
  'policyCode': {'iv': 0.0, 'woe': {1.0: 0.0}},
  'n11': {'iv': 5.669110692040425e-05,
   'woe': {0.0: -0.0013462832120869654,
    1.0: 0.0003796115927952374,
    2.0: -1.321366999272626,
    4.0: -10.126261596374201}},
  'loanAmnt_group_Bin': {'iv': 0.03343117392609739,
   'woe': {'Bin 0': -0.25565091759789366,
    'Bin 1': -0.10181829737829946,
    'Bin 2': 0.048523004375282064,
    'Bin 3': 0.15719425693473638,
    'Bin 4': 0.2554982959104215}},
  'interestRate_group_Bin': {'iv': 0.43944475156239415,
   'woe': {'Bin 0': -1.4336529379346452,
    'Bin 1': -0.47629575281922726,
    'Bin 2': 0.1260287233070282,
    'Bin 3': 0.638424679929549,
    'Bin 4': 1.1369940841694457}},
  'installment_group_Bin': {'iv': 0.027688805426894836,
   'woe': {'Bin 0': -0.29335432895221053,
    'Bin 1': -0.02248530913230762,
    'Bin 2': 0.11333264072431821}},
  'employmentTitle_group_Bin': {'iv': 0.011673058473026044,
   'woe': {'Bin 0': 0.19241479983416104,
    'Bin 1': 0.01650461957193068,
    'Bin 2': -0.18949121362830715}},
  'homeOwnership_group_Bin': {'iv': 0.0002916379612195541,
   'woe': {'Bin 0': -0.00598818429937241,
    'Bin 1': 0.04866764184572045,
    'Bin 2': 0.0582736696198456}},
  'annualIncome_group_Bin': {'iv': 0.028734272942132155,
   'woe': {'Bin 0': 0.20550314786386936,
    'Bin 1': 0.1013731532894062,
    'Bin 2': -0.029439154506365764,
    'Bin 3': -0.19611334622465693,
    'Bin 4': -0.3341722425856623}},
  'purpose_group_Bin': {'iv': 0.007398348262121109,
   'woe': {'Bin 0': 0.06057157985102101, 'Bin 1': -0.12221757368584851}},
  'postCode_group_Bin': {'iv': 0.0007570488920220754,
   'woe': {'Bin 0': -0.10132004266079044,
    'Bin 1': -0.005811282866672897,
    'Bin 2': 0.02460536621769406}},
  'regionCode_group_Bin': {'iv': 0.0003170244903853317,
   'woe': {'Bin 0': 0.010672734282877688, 'Bin 1': -0.029704933969456917}},
  'dti_group_Bin': {'iv': 0.06736335316502741,
   'woe': {'Bin 0': -0.2732685235815132,
    'Bin 1': -0.03476878794313418,
    'Bin 2': 0.13763524255784043,
    'Bin 3': 0.28987300233823476,
    'Bin 4': 0.4903505432780388}},
  'delinquency_2years_group_Bin': {'iv': 0.0022623712300652897,
   'woe': {'Bin 0': -0.012708363930036529,
    'Bin 1': 0.1475062082023903,
    'Bin 2': 0.20801887412532466}},
  'ficoRangeLow_group_Bin': {'iv': 0.12055544422234395,
   'woe': {'Bin 0': 0.28837473031500915,
    'Bin 1': -0.008500511432484521,
    'Bin 2': -0.30104870120279137,
    'Bin 3': -0.5875041637976371,
    'Bin 4': -1.0326430545002714}},
  'ficoRangeHigh_group_Bin': {'iv': 0.12055544422234395,
   'woe': {'Bin 0': 0.28837473031500915,
    'Bin 1': -0.008500511432484521,
    'Bin 2': -0.30104870120279137,
    'Bin 3': -0.5875041637976371,
    'Bin 4': -1.0326430545002714}},
  'openAcc_group_Bin': {'iv': 0.004306029382523432,
   'woe': {'Bin 0': -0.07821428563153432,
    'Bin 1': -0.019151061460234813,
    'Bin 2': 0.034864790449546074,
    'Bin 3': 0.10529612262224165,
    'Bin 4': 0.17443385861739738}},
  'pubRec_group_Bin': {'iv': 0.0012341393260821526,
   'woe': {'Bin 0': -0.0061018199179689245, 'Bin 1': 0.20227836670835725}},
  'pubRecBankruptcies_group_Bin': {'iv': 0.004386867849714986,
   'woe': {'Bin 0': -0.025551658775713103,
    'Bin 1': 0.16462934316291766,
    'Bin 2': 0.2489311629377492}},
  'revolBal_group_Bin': {'iv': 0.003941432411932272,
   'woe': {'Bin 0': 0.025043463019543088,
    'Bin 1': -0.057776412254312,
    'Bin 2': -0.2522823282700653}},
  'revolUtil_group_Bin': {'iv': 0.02315807262983133,
   'woe': {'Bin 0': -0.34972229902847746,
    'Bin 1': -0.15293900491846227,
    'Bin 2': 0.012278839217558546,
    'Bin 3': 0.09393480729600885,
    'Bin 4': 0.16029349283335198}},
  'totalAcc_group_Bin': {'iv': 0.001898851737310343,
   'woe': {'Bin 0': 0.06652798510153149,
    'Bin 1': 0.0063347829562498655,
    'Bin 2': -0.022235005488180352,
    'Bin 3': -0.04113034955049715}},
  'title_group_Bin': {'iv': 0.021017346706830586,
   'woe': {'Bin 0': 0.12026055634191922,
    'Bin 1': -0.06297069710648713,
    'Bin 2': -0.29878201272196026}},
  'n0_group_Bin': {'iv': 0.0038659161878576197,
   'woe': {'Bin 0': -0.03432653353796549, 'Bin 1': 0.11265809331135226}},
  'n1_group_Bin': {'iv': 0.01251701221843841,
   'woe': {'Bin 0': -0.11118731349483088,
    'Bin 1': -0.030385344572322486,
    'Bin 2': 0.024394768158194042,
    'Bin 3': 0.10070490376937744,
    'Bin 4': 0.24324257832089885}},
  'n2_group_Bin': {'iv': 0.03182485856762544,
   'woe': {'Bin 0': -0.25234529542718825,
    'Bin 1': -0.12039210786755326,
    'Bin 2': 0.01654240445119373,
    'Bin 3': 0.12840398734442102,
    'Bin 4': 0.30175137102433}},
  'n3_group_Bin': {'iv': 0.03182485856762544,
   'woe': {'Bin 0': -0.25234529542718825,
    'Bin 1': -0.12039210786755326,
    'Bin 2': 0.01654240445119373,
    'Bin 3': 0.12840398734442102,
    'Bin 4': 0.30175137102433}},
  'n4_group_Bin': {'iv': 0.005267251015654909,
   'woe': {'Bin 0': -0.28953419649589196,
    'Bin 1': -0.011632803273824087,
    'Bin 2': 0.016095261458987927,
    'Bin 3': 0.07691228812277608}},
  'n5_group_Bin': {'iv': 0.0002521958240056829,
   'woe': {'Bin 0': 0.025796860662542026, 'Bin 1': -0.009776426926970056}},
  'n6_group_Bin': {'iv': 0.0013155150687629844,
   'woe': {'Bin 0': -0.07315867586330163,
    'Bin 1': 0.005254489185691692,
    'Bin 2': 0.11321926249888056}},
  'n7_group_Bin': {'iv': 0.009545069117979268,
   'woe': {'Bin 0': -0.14454735339316377,
    'Bin 1': -0.024874826581771457,
    'Bin 2': 0.03959338275756879,
    'Bin 3': 0.08367884371104235,
    'Bin 4': 0.18950818931946098}},
  'n8_group_Bin': {'iv': 0.0006247355335945231,
   'woe': {'Bin 0': -0.06522263650073688, 'Bin 1': 0.009579006538587252}},
  'n9_group_Bin': {'iv': 0.030937050220452105,
   'woe': {'Bin 0': -0.20889047134965136,
    'Bin 1': -0.04280319854690665,
    'Bin 2': 0.07380942227136413,
    'Bin 3': 0.19161748950791213,
    'Bin 4': 0.3782251363238346}},
  'n10_group_Bin': {'iv': 0.008253500739303452,
   'woe': {'Bin 0': -0.29328765643532173,
    'Bin 1': -0.06153347064891405,
    'Bin 2': -0.0017968871206732596,
    'Bin 3': 0.047852298082086936,
    'Bin 4': 0.1469804159216567}},
  'n12_group_Bin': {'iv': 8.313258112458898e-05,
   'woe': {'Bin 0': -0.0005117920937828983, 'Bin 1': 0.16243538529117849}},
  'n13_group_Bin': {'iv': 0.0017296224617373412,
   'woe': {'Bin 0': -0.010273646500446446, 'Bin 1': 0.16837952088281963}},
  'n14_group_Bin': {'iv': 0.05013255836162971,
   'woe': {'Bin 0': -0.31114470867320426,
    'Bin 1': -0.15108234506634566,
    'Bin 2': 0.0032456128370830016,
    'Bin 3': 0.15067931120695519,
    'Bin 4': 0.33510301297940864}}}}
  ```
 **result**：最终的结果，长这个样子：
   ```python
 {'model_summary': <class 'statsmodels.iolib.summary.Summary'>
 """
                            Logit Regression Results                           
 ==============================================================================
 Dep. Variable:              isDefault   No. Observations:               560000
 Model:                          Logit   Df Residuals:                   559984
 Method:                           MLE   Df Model:                           15
 Date:                Thu, 04 Mar 2021   Pseudo R-squ.:                 0.08853
 Time:                        14:28:07   Log-Likelihood:            -2.5537e+05
 converged:                       True   LL-Null:                   -2.8018e+05
 Covariance Type:            nonrobust   LLR p-value:                     0.000
 ===========================================================================================
                               coef    std err          z      P>|z|      [0.025      0.975]
 -------------------------------------------------------------------------------------------
 term                        0.4856      0.010     48.088      0.000       0.466       0.505
 verificationStatus          0.2431      0.016     15.175      0.000       0.212       0.275
 loanAmnt_group_Bin          0.2776      0.041      6.815      0.000       0.198       0.357
 interestRate_group_Bin      0.1153      0.016      7.022      0.000       0.083       0.148
 installment_group_Bin       0.2646      0.039      6.728      0.000       0.187       0.342
 annualIncome_group_Bin      1.0481      0.025     42.018      0.000       0.999       1.097
 dti_group_Bin               0.3618      0.014     25.125      0.000       0.334       0.390
 ficoRangeLow_group_Bin      0.1894        nan        nan        nan         nan         nan
 ficoRangeHigh_group_Bin     0.1894        nan        nan        nan         nan         nan
 revolUtil_group_Bin        -0.2620      0.028     -9.307      0.000      -0.317      -0.207
 title_group_Bin             0.4616      0.025     18.401      0.000       0.412       0.511
 n2_group_Bin                0.0472        nan        nan        nan         nan         nan
 n3_group_Bin                0.0472        nan        nan        nan         nan         nan
 n9_group_Bin                0.2948      0.004     74.254      0.000       0.287       0.303
 n14_group_Bin               0.3586      0.018     20.174      0.000       0.324       0.393
 grade_group_Bin             0.3026      0.017     17.876      0.000       0.269       0.336
 subGrade_group_Bin          0.3198      0.014     22.258      0.000       0.292       0.348
 intercept                  -1.3887      0.004   -387.449      0.000      -1.396      -1.382
 ===========================================================================================
 """,
 'ks': 0.29957454898668295,
 'auc': 0.7067054639204269,
 'val_ks': 0.29633281875616696,
 'val_auc': 0.7043781162452405}
  ```

 **select_value_iv**：一个函数来的，不用管啦
 
 **train_data**：训练集
 
 **val_data**：测试集
 
 **y_name**：你输入的y值的变量名
 
 
 
 
 
 #### 写到我手都废了，你觉得我辛苦的话，可以关注我的公众号：**屁屁和铭仔的数据之路**
 #### 好气哦，不知道怎么贴收款码，万一你们想打赏我们两个怎么办！！！！！
