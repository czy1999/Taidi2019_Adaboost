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
策略思路：
1. 回测标的：上证50成分股
2. 回测时间段：2016-01-01 至 2019-05-30
3. 特征选择：
质量类 
    CurrentRatio    流动比率
    NetProfitRatio  销售净利率
    ROE 权益回报率
情绪类 
    VSTD10  10日成交量标准差
    TVMA20  10日平均换手率
    VR  成交量比率
成长类 
    NetProfitGrowRate   净利润增长率
常用技术指标类 
    BOLLDOWN    下轨线（布林线）指标
    MTM10   动量指标
    KDJ_D   随机指标D
    BBI 多空指数
动量类 
    BIAS20  20日乖离率
    DDI 方向标准离差指数
价值类 
    NegMktValue 流通市值
    PE  市盈率
    PS  市销率
    PB  市净率
    MktValue    总市值
每股指标类   
    BasicEPS    基本每股收益
    EPS 每股收益TTM值
特色技术指标类 
    NATR    归一化平均真实范围
    STDDEV  标准差

'''

#选取的因子列表
factor_list = ['ROE','VSTD10','TVMA20','NetProfitGrowRate','MTM10','NegMktValue','PE','PB','MktValue','NATR','Sharperatio20']

#初始化函数
def init(context):
    # 设置回测初始金额
    set_backtest(initial_cash=10000000)  
    # 注册K线数据
    reg_kdata('day', 1)  
    # 注册因子数据
    reg_factor(factor_list)
    # 设置交易日期
    days = get_trading_days('SSE', '2016-01-01', '2019-05-30')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()
    # 模型训练次数
    context.num = 0
    # 建立Adaboost模型
    context.pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
    context.hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}
    # 读取保存的模型
    #context.clf = joblib.load("model.joblib")
    context.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200,algorithm="SAMME.R",learning_rate=0.1)

# 标准化函数
def Get_BZH(list):
    L = (list - min(list))/(max(list) - min(list))
    return L

# 数据处理函数，每次交易都会运行一遍
def on_data(context):
    if dt.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:  # 调仓频率为月
        return

    # 读取因子值
    factor = get_reg_factor(context.reg_factor[0], target_indices=[x for x in range(50)], length=1, df=True).fillna(method = 'backfill', axis = 1)
    #print(factor)

    # 存储训练特征及标签样本
    FactorData = pd.DataFrame(columns=(['target_idx'] + factor_list)).set_index('target_idx') 
    for i in factor_list:
        FactorData[i]=factor[factor['factor'] == i].value.values

    # 标准化处理
    for i in factor_list:
        if i != 'Sharperatio20':
            FactorData[i]=Get_BZH(FactorData[i])


    # 收益率分类：夏普比率为正的分类为1，其他为0
    for i,v in enumerate(FactorData['Sharperatio20']):

        if(v>0):
            FactorData['Sharperatio20'][i] = 1
        else:
            FactorData['Sharperatio20'][i] = 0

    # 缺失值处理
    # 先向后取值，若仍存在Nan值，则删除该特征因子
    X_test = FactorData.drop(columns=['Sharperatio20']).fillna(method = 'ffill')
    context.y_train = FactorData['Sharperatio20'].fillna(method = 'ffill')
    X_test = X_test.dropna(axis = 1)

    # 建立标的列表，存放目标股票的序号
    buy_list = []

    # 机器学习优化
    # 第一次就取平均权重
    if context.num ==0:
        context.num = context.num + 1
        context.X_train=X_test
        # 所有特征值相加，取最大的五个股票
        temp = context.X_train.sum(axis=1).sort_values(ascending=False)
        buy_list = temp.index.values[0:5]

    # 从第二次开始，每次用新的训练集进行训练
    else:
        # Fit and tune model
        #print('X_train',context.X_train)
        #print('y_train',context.y_train)

        #开始训练本次的模型
        context.clf.fit(context.X_train, context.y_train.astype('int'))
        print("第",context.num,"次模型训练完毕")
        # clf = joblib.load('rf_regressor.pkl')
        # 储存下一次的训练集
        # joblib.dump(context.clf, "m/model.joblib")

        # 开始预测
        y_pred = context.clf.predict(X_test).tolist()

        # 将本次预测的特征作为下次训练的特征集
        context.X_train=X_test

        # 选取预测收益率为正的股票，放入buy_list中
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                buy_list.append(i)
        print("buy_list",buy_list)

    # 模型训练次数+1
    context.num = context.num + 1
    

    # 交易设置：
    #positions_long = context.account().positions['volume_long']  # 多头持仓数量
    #valid_cash = context.account(account_idx=0).cash['valid_cash'][0]  # 可用资金
    # 获取股票的权重
    #buy_value = valid_cash / (len(buy_list) + 1)  # 设置每只标的可用资金比例 + 1 防止分母为0
    #print('buyvalue',buy_value)

    # 开始交易
    positions = context.account().positions
    # 平不在标的池的股票
    for target_idx in positions.target_idx.astype(int):
        if target_idx not in buy_list:
            if positions['volume_long'].iloc[target_idx] > 0:
                order_volume(account_idx=0, target_idx=target_idx,
                             volume=int(positions['volume_long'].iloc[target_idx]),
                             side=2, position_effect=2, order_type=2, price=0)
                print('卖出', context.target_list[target_idx])

    # 获取股票的权重
    percent = 0.2
    # 买在标的池中的股票
    for target_idx in buy_list:
        context.order_id=order_target_percent(account_idx=0, target_idx=int(target_idx), 
            target_percent=percent, side=1, order_type=2,
            price=0)
        print('买入', context.target_list[target_idx])


if __name__ == '__main__':
    # 设置开始与结束时间
    begin = '2016-01-01'
    end = '2019-05-30'
    cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
    # 设置目标股票池，此处为上证50 
    sz50 = get_code_list('sz50', cons_date)[['code', 'weight']]
    targetlist = list(sz50['code'])
    # 开始回测
    run_backtest(strategy_name='Adaboost',
                 file_path='adaboost.py',
                 target_list=targetlist,
                 frequency='day',
                 fre_num=1,
                 begin_date=begin,
                 end_date=end,
                 fq=1)
