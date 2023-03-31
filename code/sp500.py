import math
import random
import pandas as pd
import numpy as np

from ArctanTransform import arctan_trans
from Effective import time_one
from EquilibriumParameter import feature_distribution
from StateParameter import state_parameter_set
import matplotlib.pyplot as plt

from ToCsv import dataToCsv
from correlation import attribution_correlation, regression_models
import time

data_raw = pd.read_csv("sp500/attribute/try1.csv")



no_stock = 504  # the number of all stock
name_stock = data_raw[:, 0]

###########################################
# setting for time period
###########################################

start = -61  # ceg ipo only 60 days
end = -1
no_date = end - start

############################################
# for choosing attributes
############################################

# attribute = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # attribute 2:18
# attribute = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

# attribute = [2,8,11,12,14,15,17,18,19]
attribute = [8]#, 11, 12, 14, 15, 17, 18]
# attribute = [19]
# attribute = [2]



no_attribute = len(attribute)
# data_attribute = data_raw[:, attribute]
############################################

############################################
# 多次循环运行时间测试
############################################
test_model_choice = 1  # if 1, no loop, 2 loop and only for select_stock_item = 2
select_stock_item = 1  # 1 for all, 2 for select order, 3 for select random， 4 for select specific stocks
loop_time_set = 100  # if 100, it means loop 100 times

if test_model_choice == 1:
    loop_time = 1
else:
    select_stock_item = 2
    loop_time = loop_time_set

select_order = 0  # if 10,it means first 10 stocks in order
time_record = []

for i in range(loop_time):
    select_order += 5

    ############################################
    # selecting stock
    ############################################

    # select_order = 50
    select_no = 100  # if 10, it means randomly choosing 10 stocks

    select_specific_stock = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  #

    if select_stock_item == 1:
        no_name = range(len(name_stock))
        # print(no_name)
        number_name = range(len(name_stock))
        print(number_name)
    elif select_stock_item == 2:
        no_name = range(select_order)
        # print(no_name)
        number_name = range(select_order)
    elif select_stock_item == 3:
        random.seed(0)
        no_name = sorted(random.sample(range(0, no_stock - 1), select_no))
        # print(no_name)
        number_name = range(select_no)  # no_name
    else:
        no_name = select_specific_stock
        number_name = range(len(select_specific_stock))

    ############################################

    date = pd.read_csv("sp500/A.csv")
    date = np.array(date)
    date = date[start:end, 0].astype(str)

    daily_price_raw = []
    daily_volume = []
    attribute_stock = []
    attribute_select = []

    for i in no_name:
        share = pd.read_csv("sp500/" + name_stock[i] + ".csv")
        stock = np.array(share)
        price = stock[start:end, 5].astype(float)  # 5 is price, 6 is the volume
        daily_price_raw.append(price)  # row is the share, column is date
        volume = stock[start:end, 6].astype(float)
        daily_volume.append(volume)  # row is the share, column is date
        attribute_raw = data_raw[i, :]
        attribute_stock.append(attribute_raw)  ####
        attribute_select_raw = data_raw[i, attribute]
        attribute_select.append(attribute_select_raw)

    attribute_stock = np.array(attribute_stock)
    attribute_select = np.array(attribute_select)

    daily_volume = list(map(list, zip(*daily_volume)))

    daily_state_raw = []

    for i in number_name:
        state = attribute_stock[i, -1] * daily_price_raw[i]
        state = np.array(state.astype(float))
        daily_state_raw.append(state)
    # dataToCsv('sp.csv', daily_state_raw, name, date, name)

    daily_state = list(map(list, zip(*daily_state_raw)))
    # daily_price = list(map(list, zip(*daily_price_raw)))
    # data1 = list(map(list, zip(*data)))

    ################################################
    ######### test     #############################
    ################################################
    # time start
    start_time = time.time()

    # test setting
    change = 0.005  # if 0.01, it means 1% up or down
    test_start = 0  # setting the t0 state (and the esps date)

    h_start = 30  # prediction start time t+h_start
    h_end = 31  # prediction end time t+h_end; last is 59, it means 60th day
    h = h_end - h_start  # setting prediction time period

    choose = 2  # choose method to calculate esps (1 is without volume, 2 is with)

    w = 0
    v = 0

    TA = []  # the total accuracy
    CCV = []  # accuracy rate

    for o in range(no_date - h_end - test_start):
        w += 1
        print(w)
        if w == 30:
            break

        # calculate sps
        spss = []
        for i in range(test_start + h_start - 1, test_start + h_end):  # test_start+h_start-1 is t_0
            sps = state_parameter_set(daily_state[i])
            sps = sps.astype(float)
            spss.append(sps)

        x_c = attribute_select
        y_c = spss[0]
        correlates = attribution_correlation(3, x_c, y_c)  # 3 is ols
        # print(correlates)

        # print(daily_state[test_start])
        # print(correlates)

        if choose == 1:
            # calculate eqs without volume
            # if without volume esps not change over time
            choice = 0
            x = 0
            for i in range(no_attribute):
                a = feature_distribution(attribute_select[:, i], 1)  # correlates[i])
                x += a
        else:
            # calculate eqs with volume
            choice = 1
            x = 0
            for i in range(no_attribute):
                a = feature_distribution(attribute_select[:, i], correlates[i])
                x += a

            volumet0 = daily_volume[test_start]
            vol = np.array(volumet0).reshape((-1, 1))
            vol_c = attribution_correlation(3, vol, y_c)
            x += feature_distribution(volumet0, vol_c)

        x += feature_distribution(spss[0], 1)
        # x1 = feature_distribution(spss[0], 1)
        # x += spss[0]

        # esps = ((x / no_attribute + choice) + 1) / 504 # 旧重构公式

        # esps = (x - np.min(x)) / (-np.min(x)*504)
        # esps = (x1 - np.min(x1)) / (-np.min(x1) * 504)
        x1 = x / no_attribute

        start_state = 1 / no_stock
        # transfor
        x1_trans = np.array(arctan_trans(x1))

        # x2 = np.asarray(x1_trans)
        # 特征值叠加后必定会出现超过100%的情况，因此使用反正切转换将 7/8修改
        x1_tan = (2 / math.pi) * start_state * x1_trans

        # esps = (x1 - np.min(x1)) / abs(np.sum(np.min(x1)))
        esps = start_state + x1_tan
        # print(np.min(esps))

        # test prediction

        base = []

        statet0 = spss[0]  # num is in abs(start:end)
        statetx = spss[1:]

        for i in number_name:
            b = (esps[i] - statet0[i]) / statet0[i]
            if b > change:
                c = 1
            elif change >= b >= -change:
                c = 2
            else:
                c = 3
            base.append(c)

        test = []
        for i in range(h):
            t = []
            s1 = statetx[i]
            for j in number_name:
                t1 = (s1[j] - statet0[j]) / statet0[j]
                if t1 > change:
                    d = 1
                elif change >= t1 >= -change:
                    d = 2
                else:
                    d = 3
                t.append(d)
            test.append(t)

        #################
        # result for accuracy ratio
        pv = []
        for i in range(h):
            n = 0
            m = 0
            p1 = test[i]
            for j in number_name:
                m += 1
                if p1[j] == base[j]:
                    n += 1
            pv.append(n / m)

        # print(pv)

        k = 0
        l = 0
        z = 0
        y = 0

        for i in range(len(pv)):
            k += 1
            if pv[i] >= 0.7:
                l += 1
            elif 0.6 <= pv[i] < 0.7:
                z += 1
            elif 0.5 <= pv[i] < 0.6:
                y += 1

        # summary
        u = (l + z + y) / k  # prediction rate >= 0.5 per test
        # u = (l + z) / k
        # u = l / k
        if u >= 0.6:  # statistics the number of prediction correct rate >= 0.6 for all test
            v += 1
        # print(v)
        # print(l / k, z / k, y / k)
        # print(u)

        accuracy_recod = []
        mm = 0
        nn = 0
        vv = 0

        for i in range(h):
            p1 = test[i]
            for j in number_name:
                if p1[j] == base[j]:
                    pa = 2
                    mm += 1
                elif p1[j] == 2:
                    pa = 1
                    vv += 1
                else:
                    pa = 0
                accuracy_recod.append(pa)
                nn += 1

        TA.append(accuracy_recod)
        cc = mm/ (nn-vv)
        CCV.append(cc)

        test_start += 1

    TA = np.array(TA)
    TA = TA.T
    np.array(TA)
    save = pd.DataFrame(TA)
    save.to_csv('TA.csv', index=False, header=False)

    CCV = np.array(CCV)

    save = pd.DataFrame(CCV)
    save.to_csv('CCV.csv', index=False, header=False)

    # print(v / w)  # prediction correct rate >= 0.7 for all test

    # print(time_one())
    # 1657267196.3012242

    end_time = time.time()  # 1657267201.6171696

    time1 = end_time - start_time
    time_record.append(time1)

    # print('spend： %s second' % (end_time - start_time))
    #
    #
    ###########################
    # result for each stock with each time point

# csv
time_record = np.array(time_record)
time_record = time_record.T
np.array(time_record)
save = pd.DataFrame(time_record, columns=['time'])
save.to_csv('time.csv', index=False, header=False)

print('finish')
