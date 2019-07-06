# -*- coding: utf-8 -*-

from atrader import *
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split    
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


'''
回测数据:上证50
回测时间:2016-01-01 到2018-09-31 
'''


def init(context):
    set_backtest(initial_cash=10000000)  # 设置回测初始信息
    reg_kdata('day', 1)  # 注册K线数据
    reg_factor(['ROE','VSTD10','TVMA20','NetProfitGrowRate','MTM10','NegMktValue','PE','PB','MktValue','NATR','Sharperatio20'])
    days = get_trading_days('SSE', '2016-01-01', '2018-09-30')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()
    context.num = 0
    context.pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
    context.hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}
    context.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200,algorithm="SAMME.R",learning_rate=0.1)
# 标准化函数
def Get_BZH(list):
    L = (list - min(list))/(max(list) - min(list))
    return L

def PD(num):
    if num>0:
        return 1
    else:
        return 0

def on_data(context):
    if dt.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:  # 调仓频率为月
        return

    # 读取因子值
    factor = get_reg_factor(context.reg_factor[0], target_indices=[x for x in range(50)], length=1, df=True)
    ROE = factor[factor['factor'] == 'ROE'].rename(columns={'value': 'ROE'}).drop('factor', axis=1).set_index('target_idx')
    VSTD10 = factor[factor['factor'] == 'VSTD10'].rename(columns={'value': 'VSTD10'}).drop('factor', axis=1).set_index('target_idx')    
    TVMA20 = factor[factor['factor'] == 'TVMA20'].rename(columns={'value': 'TVMA20'}).drop('factor', axis=1).set_index('target_idx')
    NetProfitGrowRate = factor[factor['factor'] == 'NetProfitGrowRate'].rename(columns={'value': 'NetProfitGrowRate'}).drop('factor', axis=1).set_index('target_idx')
    MTM10 = factor[factor['factor'] == 'MTM10'].rename(columns={'value': 'MTM10'}).drop('factor', axis=1).set_index('target_idx')
    NegMktValue = factor[factor['factor'] == 'NegMktValue'].rename(columns={'value': 'NegMktValue'}).drop('factor', axis=1).set_index('target_idx')    
    PE = factor[factor['factor'] == 'PE'].rename(columns={'value': 'PE'}).drop('factor', axis=1).set_index('target_idx')
    PB = factor[factor['factor'] == 'PB'].rename(columns={'value': 'PB'}).drop('factor', axis=1).set_index('target_idx')
    MktValue = factor[factor['factor'] == 'MktValue'].rename(columns={'value': 'MktValue'}).drop('factor', axis=1).set_index('target_idx')
    NATR = factor[factor['factor'] == 'NATR'].rename(columns={'value': 'NATR'}).drop('factor', axis=1).set_index('target_idx')    
    Sharperatio20 = factor[factor['factor'] == 'Sharperatio20'].rename(columns={'value': 'Sharperatio20'}).drop('factor', axis=1).set_index('target_idx')    

    
    # 标准化处理
    ROE.ROE = Get_BZH(ROE.ROE)
    VSTD10.VSTD10 = Get_BZH(VSTD10.VSTD10)
    TVMA20.TVMA20 = Get_BZH(TVMA20.TVMA20)
    NetProfitGrowRate.NetProfitGrowRate = Get_BZH(NetProfitGrowRate.NetProfitGrowRate)
    MTM10.MTM10 = Get_BZH(MTM10.MTM10)
    NegMktValue.NegMktValue = Get_BZH(NegMktValue.NegMktValue)
    PE.PE = Get_BZH(PE.PE)
    PB.PB = Get_BZH(PB.PB)
    MktValue.MktValue = Get_BZH(MktValue.MktValue)
    NATR.NATR = Get_BZH(NATR.NATR)

    X_test = pd.concat([ROE.ROE,PB.PB,PE.PE,VSTD10.VSTD10,TVMA20.TVMA20,NetProfitGrowRate.NetProfitGrowRate,MTM10.MTM10,NegMktValue.NegMktValue,MktValue.MktValue,NATR.NATR],axis=1)

    MB = []
    # 机器学习优化
    # 第一次就取平均权重
    if context.num ==0:
        context.X_train=pd.concat([ROE.ROE,PB.PB,PE.PE,VSTD10.VSTD10,TVMA20.TVMA20,NetProfitGrowRate.NetProfitGrowRate,MTM10.MTM10,NegMktValue.NegMktValue,MktValue.MktValue,NATR.NATR],axis=1)
        context.num = context.num + 1
        PE.PE = (PE.PE+ROE.ROE+VSTD10.VSTD10+TVMA20.TVMA20+NetProfitGrowRate.NetProfitGrowRate+MTM10.MTM10+NegMktValue.NegMktValue+PE.PE+PB.PB+MktValue.MktValue+NATR.NATR)/10
        PE = PE.sort_values(by="PE",ascending= False)
        MB = PE.index.values[0:5]

    # 从第二次开始，每次用新的训练集进行训练
    else:
        context.y_train=Sharperatio20.Sharperatio20

        # Fit and tune model
        test = pd.concat([context.y_train,context.X_train],axis =1)
        test.dropna(inplace=True)
        test.columns = ['SP','OE', 'PB', 'PE', 'VSTD10', 'TVMA20', 'NetProfitGrowRate', 'MTM10', 'NegMktValue', 'MktValue', 'NATR']

        context.y_train= test['SP']


        # context.y_train= [PD(x) for x in context.y_train]
        #print('ontext.y_train[0]',context.y_train.index.values)
        for i in context.y_train.index.values:
            print(i,PD(context.y_train[i]))
            context.y_train[i] = PD(context.y_train[i]) 
        print(context.y_train)
        context.X_train= test.drop('SP', axis=1)


        #context.X_train.loc[0].values
        #context.X_train = [context.X_train.loc[i].values.tolist() for i in range(50)]
        #print(context.X_train)
        context.clf.fit(context.X_train, context.y_train.astype('int'))
        print("第",context.num,"次模型训练完毕")
        #clf = joblib.load('rf_regressor.pkl')
        # 储存下一次的训练集
        context.X_train=X_test
        # 开始预测
        X_test.dropna(inplace=True)
        y_pred = context.clf.predict(X_test).tolist()

        print(y_pred)
        if '1' in y_pred:
            print("2323",y_pred.index('1'))
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                MB.append(i)
        print("MB",MB)

    context.num = context.num + 1
    


    positions = context.account().positions
    # 平不在标的池的股票
    for target_idx in positions.target_idx.astype(int):
        if target_idx not in MB:
            if positions['volume_long'].iloc[target_idx] > 0:
                order_volume(account_idx=0, target_idx=target_idx,
                             volume=int(positions['volume_long'].iloc[target_idx]),
                             side=2, position_effect=2, order_type=2, price=0)
                print('卖出', context.target_list[target_idx])

    # 获取股票的权重
    percent = 0.2
    # 买在标的池中的股票
    for target_idx in MB:
        order_target_percent(account_idx=0, target_idx=int(target_idx), target_percent=percent, side=1, order_type=2,
                             price=0)
        print('买入', context.target_list[target_idx])

if __name__ == '__main__':
    begin = '2016-01-01'
    # end = '2018-09-30'
    end = '2018-09-30'
    cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
    sz50 = get_code_list('sz50', cons_date)[['code', 'weight']]
    targetlist = list(sz50['code'])
    targetlist.append('sse.000016')
    run_backtest(strategy_name='Adaboost',
                 file_path='adaboost.py',
                 target_list=targetlist,
                 frequency='day',
                 fre_num=1,
                 begin_date=begin,
                 end_date=end,
                 fq=1)