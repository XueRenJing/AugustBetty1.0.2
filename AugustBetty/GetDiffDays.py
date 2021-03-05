


import time 
import datetime 
from dateutil import rrule

# 与当前相差天数
def get_diff_days_2_now(date_str): 
    '''
    

    Parameters
    ----------
    date_str : 一个时间点

    Returns
    -------
    diff_days : 与当前时间的差的天数

    '''
    try:
        now_time = time.localtime(time.time()) 
        compare_time = time.strptime(date_str, "%Y-%m-%d")
        # 比较日期
        date1 = datetime.datetime(compare_time[0], compare_time[1], compare_time[2])
        date2 = datetime.datetime(now_time[0], now_time[1], now_time[2])
        diff_days = (date2 - date1).days
    except:
        diff_days=None
    return diff_days
def get_diff_days_interval(date_str1,date_str2): 
    '''
    支持这种类型的时间格式2014-07-01
    
    Parameters
    ----------
    date_str : 两个时间点

    Returns
    -------
    diff_days : 两个时间点相差的时间

    '''
    # try:
    compare_time1 = time.strptime(date_str1, "%Y-%m-%d")
    compare_time2 = time.strptime(date_str2, "%Y-%m-%d")
    # 比较日期
    date1 = datetime.datetime(compare_time1[0], compare_time1[1], compare_time1[2])
    date2 = datetime.datetime(compare_time2[0], compare_time2[1], compare_time2[2])
    diff_days = (date2 - date1).days
    # except:
    #     diff_days=None
    return diff_days
if __name__ == '__main__':
    train_data["diff_days"]=train_data["issueDate"].map(lambda x:get_diff_days_2_now(x))
    train_data["diff_month"]=train_data["diff_days"].map(lambda x : int(x/30)) 
    train_data["diff_week"]=train_data["diff_days"].map(lambda x : int(x/7)) 
    train_data["diff_year"]=train_data["diff_days"].map(lambda x : x/365)
    
    
    ff=train_data.apply(lambda x:get_diff_days_interval(x["issueDate"],x["issueDate"]),axis=1)

