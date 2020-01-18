from alpha_vantage.timeseries import TimeSeries
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import csv
from numpy import *
import time as T
import calendar
import datetime
from pytz import timezone


# predict data
def predict_intraday_data():
    # Importing the dataset
    dataset = pd.read_csv('dayily_data_test.csv')

    y = dataset.iloc[:, 4].values
    L = len(y)
    arr = np.arange(0, L)  # there is 390 more alemtns !!
    X = arr.reshape(len(arr), 1)

    arr1 = np.arange(0, 390)
    X_test = arr1.reshape(len(arr1), 1)

    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X, y)

    y_pred = regressor.predict(X_test)

    return y_pred

def predict_dayily_data():
    # Importing the dataset
    dataset = pd.read_csv('dayily_data_test.csv')
    row_mun = len(dataset.iloc[:, 4].values)
    row_mun = row_mun - 310  # 310 original #50 is better
    y = dataset.iloc[row_mun:, 4].values
    L = len(y)
    # print("L is :\n" ,L)
    arr = np.arange(0, L)
    X = arr.reshape(len(arr), 1)

    arr1 = np.arange(0, 1)
    X_test = arr1.reshape(len(arr1), 1)

    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X, y)

    # Predicting a new result
    y_pred = regressor.predict(X_test)
    return y_pred

def predict_dayily_low_data():
    # Importing the dataset
    dataset = pd.read_csv('high_low.csv')
    row_mun = len(dataset.iloc[:, 3].values)
    row_mun = row_mun - 310  # 310 original #50 is better
    y = dataset.iloc[row_mun:, 3].values
    L = len(y)
    # print("L is :\n" ,L)
    arr = np.arange(0, L)
    X = arr.reshape(len(arr), 1)

    arr1 = np.arange(0, 1)
    X_test = arr1.reshape(len(arr1), 1)

    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X, y)

    # Predicting a new result
    y_pred = regressor.predict(X_test)
    return y_pred

def predict_dayily_high_data():
    # Importing the dataset
    dataset = pd.read_csv('high_low.csv')
    row_mun = len(dataset.iloc[:, 2].values)
    row_mun = row_mun - 310  # 310 original #50 is better
    y = dataset.iloc[row_mun:, 2].values
    L = len(y)
    # print("L is :\n" ,L)
    arr = np.arange(0, L)
    X = arr.reshape(len(arr), 1)

    arr1 = np.arange(0, 1)
    X_test = arr1.reshape(len(arr1), 1)

    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X, y)

    # Predicting a new result
    y_pred = regressor.predict(X_test)
    return y_pred

# Get data
def get_intraday_data(stock_code):
    ts = TimeSeries(key='API KEY', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=stock_code, interval='1min', outputsize='full')
    return data

def get_dayily_data(stock_code):
    ts = TimeSeries(key='API KEY', output_format='pandas')
    data_day, dat_t = ts.get_daily(symbol=stock_code, outputsize='full')
    return data_day

# Time index dictionary
time_dic = {'0': '9:30', '1': '9:31', '2': '9:32', '3': '9:33', '4': '9:34', '5': '9:35', '6': '9:36', '7': '9:37',
            '8': '9:38', '9': '9:39',
            '10': '9:40', '11': '9:41', '12': '9:42', '13': '9:43', '14': '9:44', '15': '9:45', '16': '9:46',
            '17': '9:47', '18': '9:48', '19': '9:49'
    , '20': '9:50', '21': '9:51', '22': '9:52', '23': '9:53', '24': '9:54', '25': '9:55', '26': '9:56', '27': '9:57',
            '28': '9:58', '29': '9:59'
    , '30': '10:00', '31': '10:01', '32': '10:02', '33': '10:03', '34': '10:04', '35': '10:05', '36': '10:06',
            '37': '10:07', '38': '10:08', '39': '10:09'
    , '40': '10:10', '41': '10:11', '42': '10:12', '43': '10:13', '44': '10:14', '45': '10:15', '46': '10:16',
            '47': '10:17', '48': '10:18', '49': '10:19',
            '50': '10:20', '51': '10:21', '52': '10:22', '53': '10:23', '54': '10:24', '55': '10:25', '56': '10:26',
            '57': '10:27', '58': '10:28', '59': '10:29',
            '60': '10:30', '61': '10:31', '62': '10:32', '63': '10:33', '64': '10:34', '65': '10:35', '66': '10:36',
            '67': '10:37', '68': '10:38', '69': '10:39',
            '70': '10:40', '71': '10:41', '72': '10:42', '73': '10:43', '74': '10:44', '75': '10:45', '76': '10:46',
            '77': '10:47', '78': '10:48', '79': '10:49'
    , '80': '10:50', '81': '10:51', '82': '10:52', '83': '10:53', '84': '10:54', '85': '10:55', '86': '10:56',
            '87': '10:57', '88': '10:58', '89': '10:59'
    , '90': '11:00', '91': '11:01', '92': '11:02', '93': '11:03', '94': '11:04', '95': '11:05', '96': '11:06',
            '97': '11:07', '98': '11:08', '99': '11:09'
    , '100': '11:10', '101': '11:11', '102': '11:12', '103': '11:13', '104': '11:14', '105': '11:15', '106': '11:16',
            '107': '11:17', '108': '11:18', '109': '11:19',
            '110': '11:20', '111': '11:21', '112': '11:22', '113': '11:23', '114': '11:24', '115': '11:25',
            '116': '11:26', '117': '11:27', '118': '11:28', '119': '11:29',
            '120': '11:30', '121': '11:31', '122': '11:32', '123': '11:33', '124': '11:34', '125': '11:35',
            '126': '11:36', '127': '11:37', '128': '11:38', '129': '11:39',
            '130': '11:40', '131': '11:41', '132': '11:42', '133': '11:43', '134': '11:44', '135': '11:45',
            '136': '11:46', '137': '11:47', '138': '11:48', '139': '11:49'
    , '140': '11:50', '141': '11:51', '142': '11:52', '143': '11:53', '144': '11:54', '145': '11:55', '146': '11:56',
            '147': '11:57', '148': '11:58', '149': '11:59'
    , '150': '12:00', '151': '12:01', '152': '12:02', '153': '12:03', '154': '12:04', '155': '12:05', '156': '12:06',
            '157': '12:07', '158': '12:08', '159': '12:09'
    , '160': '12:10', '161': '12:11', '162': '12:12', '163': '12:13', '164': '12:14', '165': '12:15', '166': '12:16',
            '167': '12:17', '168': '12:18', '169': '12:19',
            '170': '12:20', '171': '12:21', '172': '12:22', '173': '12:23', '174': '12:24', '175': '12:25',
            '176': '12:26', '177': '12:27', '178': '10:28', '179': '10:29',
            '180': '12:30', '181': '12:31', '182': '12:32', '183': '12:33', '184': '12:34', '185': '12:35',
            '186': '12:36', '187': '12:37', '188': '12:38', '189': '12:39',
            '190': '12:40', '191': '12:41', '192': '12:42', '193': '12:43', '194': '12:44', '195': '12:45',
            '196': '12:46', '197': '12:47', '198': '12:48', '199': '12:49'
    , '200': '12:50', '201': '12:51', '202': '12:52', '203': '12:53', '204': '12:54', '205': '12:55', '206': '12:56',
            '207': '12:57', '208': '12:58', '209': '12:59'
    , '210': '13:00', '211': '13:01', '212': '13:02', '213': '13:03', '214': '13:04', '215': '13:05', '216': '13:06',
            '217': '13:07', '218': '13:08', '219': '13:09'
    , '220': '13:10', '221': '13:11', '222': '13:12', '223': '13:13', '224': '13:14', '225': '13:15', '226': '13:16',
            '227': '13:17', '228': '13:18', '229': '13:19',
            '230': '13:20', '231': '13:21', '232': '13:22', '233': '13:23', '234': '13:24', '235': '13:25',
            '236': '13:26', '237': '13:27', '238': '13:28', '239': '13:29',
            '240': '13:30', '241': '13:31', '242': '13:32', '243': '13:33', '244': '13:34', '245': '13:35',
            '246': '11:36', '247': '13:37', '248': '13:38', '249': '13:39'
    , '250': '13:40', '251': '13:41', '252': '13:42', '253': '13:43', '254': '13:44', '255': '13:45', '256': '13:46',
            '257': '13:47', '258': '13:48', '259': '13:49',
            '260': '13:50', '261': '13:51', '262': '13:52', '263': '13:53', '264': '13:54', '265': '13:55',
            '266': '13:56', '267': '13:57', '268': '13:58', '269': '13:59',
            '270': '14:00', '271': '14:01', '272': '14:02', '273': '14:03', '274': '14:04', '275': '14:05',
            '276': '14:06', '277': '14:07', '278': '14:08', '279': '14:09',
            '280': '14:10', '281': '14:11', '282': '14:12', '283': '14:13', '284': '14:14', '285': '14:15',
            '286': '14:16', '287': '14:17', '288': '14:18', '289': '14:19'
    , '290': '14:20', '291': '14:21', '292': '14:22', '293': '14:23', '294': '14:24', '295': '14:25', '296': '14:26',
            '297': '14:27', '298': '14:28', '299': '14:29'
    , '300': '14:30', '301': '14:31', '302': '14:32', '303': '14:33', '304': '14:34', '305': '14:35', '306': '14:36',
            '307': '14:37', '308': '14:38', '309': '14:39'
    , '310': '14:40', '311': '14:41', '312': '14:42', '313': '14:43', '314': '14:44', '315': '14:45', '316': '14:46',
            '317': '14:47', '318': '14:48', '319': '14:49',
            '320': '14:50', '321': '14:51', '322': '14:52', '323': '14:53', '324': '14:54', '325': '14:55',
            '326': '14:56', '327': '14:57', '328': '14:58', '329': '14:59',
            '330': '15:00', '331': '15:01', '332': '15:02', '333': '15:03', '334': '15:04', '335': '15:05',
            '336': '15:06', '337': '15:07', '338': '15:08', '339': '15:09',
            '340': '15:10', '341': '915:11', '342': '15:12', '343': '15:13', '344': '15:14', '345': '15:15',
            '346': '15:16', '347': '15:17', '348': '15:18', '349': '15:19'
    , '350': '15:20', '351': '15:21', '352': '15:22', '353': '15:23', '354': '15:24', '355': '15:25', '356': '15:26',
            '357': '15:27', '358': '15:28', '359': '15:29'
    , '360': '15:30', '361': '15:31', '362': '15:32', '363': '15:33', '364': '15:34', '365': '15:35', '366': '15:36',
            '367': '15:37', '368': '15:38', '369': '15:39'
    , '370': '15:40', '371': '15:41', '372': '15:42', '373': '15:43', '374': '15:44', '375': '15:45', '376': '15:46',
            '377': '15:47', '378': '15:48', '379': '15:49',
            '380': '15:50', '381': '15:51', '382': '15:52', '383': '15:53', '384': '15:54', '385': '15:55',
            '386': '15:56', '387': '15:57', '388': '15:58', '389': '15:59',
            '390': '16:00'}


# Sequence detection
def Sequence_detection(data, stock_name):
    L = len(data)
    start_flag = 0
    end_flag = 0
    counter = 0
    buy_flag = 0
    temp_matrix = [['sell/buy ', 'estimated price', 'index']]
    data_list = list(data)
    for i in range(1, L):
        if ((float(data_list[i]) > float(data_list[i - 1])) or (float(data_list[i]) == float(data_list[i - 1]))):
            if (start_flag == 0):
                li1 = []
                buy_index = i - 1
                buy_flag = 1
                li1.append('buy')
                li1.append(data_list[i - 1])
                li1.append(i - 1)
                temp_matrix = append(temp_matrix, [li1], 0)
                start_flag = 1
        else:
            if (start_flag == 1):
                buy_flag = 0
                sell_index = i - 1
                end_flag = 1
                li1 = []
                li1.append('sell')
                li1.append(data_list[i - 1])
                li1.append(i - 1)
                temp_matrix = append(temp_matrix, [li1], 0)

        if (start_flag == 1 and end_flag == 1):
            start_flag = 0
            end_flag = 0

    if (buy_flag == 1):
        li1 = []
        li1.append('sell')
        li1.append(data_list[-1])
        li1.append(L - 1)
        temp_matrix = append(temp_matrix, [li1], 0)

    # print (temp_matrix , "\n \n")

    temp_vector = [['buy index ', ' sell index ', 'percentage', 'stock name']]

    for i in range(1, len(temp_matrix), 2):
        list2 = []
        list2.append(temp_matrix[i][2])
        list2.append(temp_matrix[i + 1][2])
        #number = float(((float(float(temp_matrix[i + 1][1]) / float(temp_matrix[i][1]))) - 1) * 100)
        number = (100 -(float(float(float(temp_matrix[i][1])/ float(temp_matrix[i + 1][1]))* 100)))
        list2.append(number)
        list2.append(stock_name)
        if (int(list2[2]) != int(0)):
            temp_vector = append(temp_vector, [list2], 0)

    return temp_vector[1:]

# Set schedule
def Schedule(vector_matrix):
    schedule_mat = [['buy_index', 'sell index', 'percentage', 'stock_name']]

    while (True):
        if (len(vector_matrix) == 0):
            break
        min = int(vector_matrix[0][0])
        index = 0
        for i in range(1, len(vector_matrix)):
            if (int(min) > int(vector_matrix[i][0])):
                min = vector_matrix[i][0]
                index = i

        schedule_mat = append(schedule_mat, [vector_matrix[index]], 0)
        vector_matrix = np.delete(vector_matrix, index, 0)

    schedule_mat_2 = [['buy_index', 'sell index', 'percentage', 'stock_name']]

    for i in range(1, len(schedule_mat)):
        if (float(schedule_mat[i][2]) < 10):
            schedule_mat_2 = append(schedule_mat_2, [schedule_mat[i]], 0)

    """
    new_schedule_mat = [['buy_index', 'sell index' , 'percentage', 'stock_name']]

    for row in range(0 , len(schedule_mat)):
        if (schedule_mat[row][0] == schedule_mat[row-1][0]):
            temp_row = new_schedule_mat[-1]
            if (((temp_row[0])) == ((schedule_mat[row][0]))):
                if (float(schedule_mat[row][2]) > float(temp_row[2])):
                    new_schedule_mat = append(new_schedule_mat, [schedule_mat[row]], 0)
                #else:
                 #   new_schedule_mat = append(new_schedule_mat, [temp_row], 0)
        else:
             new_schedule_mat = append(new_schedule_mat, [schedule_mat[row]], 0)


    #print("new_schedule_mat : " ,new_schedule_mat , "\n")


    new_schedule_mat_1 = [['buy_index', 'sell index' , 'percentage', 'stock_name']]

    if(int(new_schedule_mat[1][1]) < int(new_schedule_mat[2][0])):
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[1]], 0)
    else:
        if (float(new_schedule_mat[1][2]) < float(new_schedule_mat[2][2])):
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[2]], 0)
        else:
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[1]], 0)

    j_index = 1

    for row in range(2 , len(new_schedule_mat)):
        if ((int(new_schedule_mat_1[j_index][1]) < int(new_schedule_mat[row][0]))): #or (int(new_schedule_mat_1[j_index][1]) == int(new_schedule_mat[row][0]))):
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[row]], 0)
            j_index += 1

        else:
            if (float(new_schedule_mat_1[j_index][2]) < (float(new_schedule_mat[row][2]))):
                new_schedule_mat_1 = np.delete(new_schedule_mat_1 ,j_index , 0)
                new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[row]], 0)


    print(" \n \n new_schedule_mat_1 : ",new_schedule_mat_1[1:] , "\n")
    return new_schedule_mat_1[1:]


    """

    new_schedule_mat = [['buy_index', 'sell index', 'percentage', 'stock_name']]

    for row in range(1, len(schedule_mat_2)):
        if (schedule_mat_2[row][0] == schedule_mat_2[row - 1][0]):
            temp_row = new_schedule_mat[-1]
            if (((temp_row[0])) == ((schedule_mat_2[row][0]))):
                if (float(schedule_mat_2[row][2]) > float(temp_row[2])):
                    new_schedule_mat = append(new_schedule_mat, [schedule_mat_2[row]], 0)
                # else:
                #   new_schedule_mat = append(new_schedule_mat, [temp_row], 0)
        else:
            new_schedule_mat = append(new_schedule_mat, [schedule_mat_2[row]], 0)

    # print("new_schedule_mat : " ,new_schedule_mat , "\n")

    new_schedule_mat_1 = [['buy_index', 'sell index', 'percentage', 'stock_name']]

    if (int(new_schedule_mat[1][1]) < int(new_schedule_mat[2][0])):
        new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[1]], 0)
    else:
        if (float(new_schedule_mat[1][2]) < float(new_schedule_mat[2][2])):
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[2]], 0)
        else:
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[1]], 0)

    j_index = 1

    for row in range(2, len(new_schedule_mat)):
        if ((int(new_schedule_mat_1[j_index][1]) < int(new_schedule_mat[row][
                                                           0]))):  # or (int(new_schedule_mat_1[j_index][1]) == int(new_schedule_mat[row][0]))):
            new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[row]], 0)
            j_index += 1

        else:
            if (float(new_schedule_mat_1[j_index][2]) < (float(new_schedule_mat[row][2]))):
                new_schedule_mat_1 = np.delete(new_schedule_mat_1, j_index, 0)
                new_schedule_mat_1 = append(new_schedule_mat_1, [new_schedule_mat[row]], 0)

    # print("j_index : ",j_index)
    return new_schedule_mat_1[1:]

def stock_to_trade(stock_list):
    counter = 0
    counter_2 = 0
    vector = [['stock name', 'Percent increase']]
    print("Enter stock_to_trade function \n")
    data = []
    for stock in range(0, len(stock_list)):

        if (counter == 7):
            counter = 0
            T.sleep(30)

        # if (counter_2 == 10):
        #   counter_2 = 0
        #  T.sleep(4)

        if (((int(stock) % 4) == 0)):
            T.sleep(60)

        print("number of stock ", stock)
        temp_li = []
        temp_li.append(stock_list[stock])
        try:
            data = get_intraday_data(stock_list[stock])
        except ValueError:
            print("stock number {} error ".format(stock))
            print("stock name :", stock_list[stock])
        except KeyError:
            print("KeyError with stock number {} , stock name {}".format(stock, stock_list[stock]))
            continue

        T.sleep(4)

        data.to_csv("high_low.csv")
        counter += 1
        # counter_2+=1
        low = predict_dayily_low_data()
        high = predict_dayily_high_data()
        temp_li.append(float(high / low))
        vector = append(vector, [temp_li], 0)

    vector = np.delete(vector, 0, 0)
    # print(" vector after delete: \n" , vector ,"\n")

    V = list(vector)

    V.sort(key=lambda x: x[1])

    # T.sleep(60)
    print("Exit stock_to_trade function \n")
    return V

def get_time_from_index(schedule_mat):
    time_dic = {'0': '9:30', '1': '9:31', '2': '9:32', '3': '9:33', '4': '9:34', '5': '9:35', '6': '9:36',
                '7': '9:37', '8': '9:38', '9': '9:39',
                '10': '9:40', '11': '9:41', '12': '9:42', '13': '9:43', '14': '9:44', '15': '9:45',
                '16': '9:46', '17': '9:47', '18': '9:48', '19': '9:49'
        , '20': '9:50', '21': '9:51', '22': '9:52', '23': '9:53', '24': '9:54', '25': '9:55', '26': '9:56',
                '27': '9:57', '28': '9:58', '29': '9:59'
        , '30': '10:00', '31': '10:01', '32': '10:02', '33': '10:03', '34': '10:04', '35': '10:05',
                '36': '10:06', '37': '10:07', '38': '10:08', '39': '10:09'
        , '40': '10:10', '41': '10:11', '42': '10:12', '43': '10:13', '44': '10:14', '45': '10:15',
                '46': '10:16', '47': '10:17', '48': '10:18', '49': '10:19',
                '50': '10:20', '51': '10:21', '52': '10:22', '53': '10:23', '54': '10:24', '55': '10:25',
                '56': '10:26', '57': '10:27', '58': '10:28', '59': '10:29',
                '60': '10:30', '61': '10:31', '62': '10:32', '63': '10:33', '64': '10:34', '65': '10:35',
                '66': '10:36', '67': '10:37', '68': '10:38', '69': '10:39',
                '70': '10:40', '71': '10:41', '72': '10:42', '73': '10:43', '74': '10:44', '75': '10:45',
                '76': '10:46', '77': '10:47', '78': '10:48', '79': '10:49'
        , '80': '10:50', '81': '10:51', '82': '10:52', '83': '10:53', '84': '10:54', '85': '10:55',
                '86': '10:56', '87': '10:57', '88': '10:58', '89': '10:59'
        , '90': '11:00', '91': '11:01', '92': '11:02', '93': '11:03', '94': '11:04', '95': '11:05',
                '96': '11:06', '97': '11:07', '98': '11:08', '99': '11:09'
        , '100': '11:10', '101': '11:11', '102': '11:12', '103': '11:13', '104': '11:14', '105': '11:15',
                '106': '11:16', '107': '11:17', '108': '11:18', '109': '11:19',
                '110': '11:20', '111': '11:21', '112': '11:22', '113': '11:23', '114': '11:24', '115': '11:25',
                '116': '11:26', '117': '11:27', '118': '11:28', '119': '11:29',
                '120': '11:30', '121': '11:31', '122': '11:32', '123': '11:33', '124': '11:34', '125': '11:35',
                '126': '11:36', '127': '11:37', '128': '11:38', '129': '11:39',
                '130': '11:40', '131': '11:41', '132': '11:42', '133': '11:43', '134': '11:44', '135': '11:45',
                '136': '11:46', '137': '11:47', '138': '11:48', '139': '11:49'
        , '140': '11:50', '141': '11:51', '142': '11:52', '143': '11:53', '144': '11:54', '145': '11:55',
                '146': '11:56', '147': '11:57', '148': '11:58', '149': '11:59'
        , '150': '12:00', '151': '12:01', '152': '12:02', '153': '12:03', '154': '12:04', '155': '12:05',
                '156': '12:06', '157': '12:07', '158': '12:08', '159': '12:09'
        , '160': '12:10', '161': '12:11', '162': '12:12', '163': '12:13', '164': '12:14', '165': '12:15',
                '166': '12:16', '167': '12:17', '168': '12:18', '169': '12:19',
                '170': '12:20', '171': '12:21', '172': '12:22', '173': '12:23', '174': '12:24', '175': '12:25',
                '176': '12:26', '177': '12:27', '178': '10:28', '179': '10:29',
                '180': '12:30', '181': '12:31', '182': '12:32', '183': '12:33', '184': '12:34', '185': '12:35',
                '186': '12:36', '187': '12:37', '188': '12:38', '189': '12:39',
                '190': '12:40', '191': '12:41', '192': '12:42', '193': '12:43', '194': '12:44', '195': '12:45',
                '196': '12:46', '197': '12:47', '198': '12:48', '199': '12:49'
        , '200': '12:50', '201': '12:51', '202': '12:52', '203': '12:53', '204': '12:54', '205': '12:55',
                '206': '12:56', '207': '12:57', '208': '12:58', '209': '12:59'
        , '210': '13:00', '211': '13:01', '212': '13:02', '213': '13:03', '214': '13:04', '215': '13:05',
                '216': '13:06', '217': '13:07', '218': '13:08', '219': '13:09'
        , '220': '13:10', '221': '13:11', '222': '13:12', '223': '13:13', '224': '13:14', '225': '13:15',
                '226': '13:16', '227': '13:17', '228': '13:18', '229': '13:19',
                '230': '13:20', '231': '13:21', '232': '13:22', '233': '13:23', '234': '13:24', '235': '13:25',
                '236': '13:26', '237': '13:27', '238': '13:28', '239': '13:29',
                '240': '13:30', '241': '13:31', '242': '13:32', '243': '13:33', '244': '13:34', '245': '13:35',
                '246': '11:36', '247': '13:37', '248': '13:38', '249': '13:39'
        , '250': '13:40', '251': '13:41', '252': '13:42', '253': '13:43', '254': '13:44', '255': '13:45',
                '256': '13:46', '257': '13:47', '258': '13:48', '259': '13:49',
                '260': '13:50', '261': '13:51', '262': '13:52', '263': '13:53', '264': '13:54', '265': '13:55',
                '266': '13:56', '267': '13:57', '268': '13:58', '269': '13:59',
                '270': '14:00', '271': '14:01', '272': '14:02', '273': '14:03', '274': '14:04', '275': '14:05',
                '276': '14:06', '277': '14:07', '278': '14:08', '279': '14:09',
                '280': '14:10', '281': '14:11', '282': '14:12', '283': '14:13', '284': '14:14', '285': '14:15',
                '286': '14:16', '287': '14:17', '288': '14:18', '289': '14:19'
        , '290': '14:20', '291': '14:21', '292': '14:22', '293': '14:23', '294': '14:24', '295': '14:25',
                '296': '14:26', '297': '14:27', '298': '14:28', '299': '14:29'
        , '300': '14:30', '301': '14:31', '302': '14:32', '303': '14:33', '304': '14:34', '305': '14:35',
                '306': '14:36', '307': '14:37', '308': '14:38', '309': '14:39'
        , '310': '14:40', '311': '14:41', '312': '14:42', '313': '14:43', '314': '14:44', '315': '14:45',
                '316': '14:46', '317': '14:47', '318': '14:48', '319': '14:49',
                '320': '14:50', '321': '14:51', '322': '14:52', '323': '14:53', '324': '14:54', '325': '14:55',
                '326': '14:56', '327': '14:57', '328': '14:58', '329': '14:59',
                '330': '15:00', '331': '15:01', '332': '15:02', '333': '15:03', '334': '15:04', '335': '15:05',
                '336': '15:06', '337': '15:07', '338': '15:08', '339': '15:09',
                '340': '15:10', '341': '15:11', '342': '15:12', '343': '15:13', '344': '15:14', '345': '15:15',
                '346': '15:16', '347': '15:17', '348': '15:18', '349': '15:19'
        , '350': '15:20', '351': '15:21', '352': '15:22', '353': '15:23', '354': '15:24', '355': '15:25',
                '356': '15:26', '357': '15:27', '358': '15:28', '359': '15:29'
        , '360': '15:30', '361': '15:31', '362': '15:32', '363': '15:33', '364': '15:34', '365': '15:35',
                '366': '15:36', '367': '15:37', '368': '15:38', '369': '15:39'
        , '370': '15:40', '371': '15:41', '372': '15:42', '373': '15:43', '374': '15:44', '375': '15:45',
                '376': '15:46', '377': '15:47', '378': '15:48', '379': '15:49',
                '380': '15:50', '381': '15:51', '382': '15:52', '383': '15:53', '384': '15:54', '385': '15:55',
                '386': '15:56', '387': '15:57', '388': '15:58', '389': '15:59',
                '390': '16:00'}
    for i in range(0, len(schedule_mat)):
        for key, value in time_dic.items():

            if (key == schedule_mat[i][0]):
                schedule_mat[i][0] = value

            if (key == schedule_mat[i][1]):
                schedule_mat[i][1] = value

    return schedule_mat

def get_units(stock_name):
    balance = barak.balance
    data = get_intraday_data(stock_name)
    data.to_csv('get_units.csv')
    dataset = pd.read_csv('get_units.csv')
    X = dataset.iloc[:, 4].values
    X_list = list(X)
    price = X_list[-1]
    units = int(balance / price)
    return units

def get_price(stock_name):
    data = get_intraday_data(stock_name)
    data.to_csv('get_price.csv')
    dataset = pd.read_csv('get_price.csv')
    X = dataset.iloc[:, 4].values
    X_list = list(X)
    price = X_list[-1]
    return price


temp_stock_list = [ 'SP', 'TNDM', 'NTRA', 'YELP', 'FN', 'STAA', 'GRVY', 'APTI', 'PCRX', 'REXR', 'OSTK', 'PKBK' ,'MMYT', 'WWW', 'NHTC', 'LPSN', 'IIVI',
                     'ALKS', 'RCKY', 'VSEC', 'AIMT', 'FE', 'HVT', 'OLN', 'TTEC', 'FIVN', 'JD', 'ARNA', 'HIIQ', 'TAL', 'CAI', 'PTLA', 'QUAD', 'DBX',
                     'MT', 'QSII', 'EVTC', 'MLI', 'HURC', 'XNCR', 'AAN' , 'CCU', 'ALTR', 'TEO', 'MDC', 'DHI', 'NCOM', 'OPY', 'TBK', 'MTOR', 'THR', 'ALRM',
                     'I', 'CUTR', 'VNOM', 'AJRD', 'EHTH', 'PETS', 'GLF', 'FDP', 'GKOS', 'TBPH', 'HCI', 'TWTR', 'CPB', 'UPLD', 'SANM', 'INT', 'GDS',
                     'ENVA', 'GBT', 'APAM', 'ETSY', 'PTCT', 'BKU', 'ACHC', 'NANO', 'IRMD', 'RM', 'KFRC', 'SSYS', 'BCOR', 'TSBK', 'PCMI' ,'TGE' , 'UEIC' ,'NRIM'
                     'QURE' ,'CEQP' , 'GBCI'  ,'YEXT' ,'ACIW' , 'EPD' ,'HTH' ,'TISI' ,'UCBI' ,'CG' ,'BLL' ,'OMER' ,'GWB' ,'RVI' ,'COOL' ,'LMNX' ,'BNFT' ,'CSFL' ,
                     'NS' , 'KTWO' ,'GGAL' ,'NATI' ,'NBHC' ,'DQ' ,'BAC' ,'BAND' , 'ACER' ,'SEND' ,'HUYA' ,'AYX' ,'HCM' ,'FSCT' ,'T']

get_ready_flag = 0
units = 0
buy_flag = 0
sell_flag = 0
buy_price = 0
sell_price = 0
real_result = [['buy time', 'sell time', 'pure profit']]
do_stuff_flag = 0
schedule_mat_2 = [[]]
start_work_time = datetime.time(9, 30, 00)
end_work_time = datetime.time(16, 00, 00)
start_time = datetime.time(4, 00, 00)
end_time = datetime.time(9, 20, 00)
Sequence_vector = [['buy index ', ' sell index ', 'percentage', 'stock name']]
data1 = []


while (True):
    now = datetime.datetime.now().astimezone(timezone('America/New_York')).today().weekday()
    day = calendar.day_name[now]
    time1 = datetime.datetime.now().astimezone(timezone('America/New_York')).time()

    if (day == 'Saturday' or day == 'Sunday'):
        print("The stock exchange is closed on the weekend Please try again on Monday \n")
        quit()

    else:
        while(True):
            time1 = datetime.datetime.now().astimezone(timezone('America/New_York')).time()
            print("The stock exchange is open begging stage 1 \n")
            # while True:
            while ((time1 >= start_time) or (time1 <= end_time) and (get_ready_flag == 0)):  # before the day begins
                stock_to_trade_list = stock_to_trade(temp_stock_list)
                L = len(stock_to_trade_list)
                number = L - 50
                stock_to_trade_list2 = stock_to_trade_list[number:]  # get ths highest 50 stocks
                list1 = []
                for i in range(0, len(stock_to_trade_list2)):
                    list1.append(stock_to_trade_list2[i][0])  # create list of 50 stock names

                counter = 0
                counter_2 = 0
                for stock in range(0, len(list1)):  # geting stock preduct data + sequence detection

                    if (counter == 7):
                        counter = 0
                        T.sleep(30)

                    if ((int(stock) % 4) == 0):
                        T.sleep(60)

                    print("number of stock ", stock)
                    try:
                        data1 = get_intraday_data(list1[stock])
                    except ValueError:
                        print("stock number {} error ".format(stock))
                        print("stock name :", list1[stock])
                        continue
                    except KeyError:
                        print("KeyError with stock number {} , stock name {}".format(stock, list1[stock]))
                        continue
                    T.sleep(4)
                    data1.to_csv('dayily_data_test.csv')
                    counter += 1
                    # counter_2+=1
                    y_pred = predict_intraday_data()
                    vector = Sequence_detection(y_pred, list1[stock])
                    for j in range(0, len(vector)):
                        Sequence_vector = append(Sequence_vector, [vector[j]], 0)

                get_ready_flag = 1
                do_stuff_flag = 1
                break



            if (do_stuff_flag == 1):
                print("stage 1 completed please check your mail \n")
                do_stuff_flag = 0
                schedule_mat = Schedule(Sequence_vector[1:])  # getting schedule for the day
                schedule_mat_2 = get_time_from_index(schedule_mat)

                print(schedule_mat_2)
                # sending email with file
                # schedule_mat_2.to_csv('day_schedule_predicted.csv')
                with open('day_schedule_predicted.csv', "w") as f:
                    wr = csv.writer(f)
                    wr.writerows(schedule_mat_2)
                f.close()
                barak.send_mail_with_file(file_name='day_schedule_predicted.csv', Subject="day schedule predicted csv file",
                                          Body="hi please don't delete this file ")

            time1 = datetime.datetime.now().astimezone(timezone('America/New_York')).time()

            if ((time1 >= start_work_time)):
                print("Workday started, begging stage 2  \n")

            counter_3 = 0
            while ((time1 >= start_work_time) or (time1 <= end_work_time) and get_ready_flag == 1):  # work day
                if(counter_3 == 4):
                    T.sleep(30)

                time1 = datetime.datetime.now().astimezone(timezone('America/New_York')).time()

                if ((((datetime.datetime.strptime(schedule_mat_2[0][0], '%H:%M').time()) <= (time1))) and (
                        buy_flag == 0)):  # buy
                    buy_flag = 1
                    units = get_units(schedule_mat_2[0][3])
                    buy_price = get_price(schedule_mat_2[0][3])
                    counter_3+=1
                    barak.buy_stocks(stock_name=schedule_mat_2[0][3], units=units)

                if ((((datetime.datetime.strptime(schedule_mat_2[0][1], '%H:%M').time()) <= (time1))) and (
                        buy_flag == 1)):  # sell
                    sell_price = get_price(schedule_mat_2[0][3])
                    counter_3 += 1
                    sell_flag = 1
                    barak.sell_stocks(stock_name=schedule_mat_2[0][3], units=units)

                if ((buy_flag == 1) and (sell_flag == 1)):
                    list1 = []
                    list1.append(schedule_mat_2[0][0])  # buy time
                    list1.append(schedule_mat_2[0][1])  # sell time
                    list1.append((sell_price - buy_price) * units)
                    real_result = append(real_result, [list1], 0)
                    buy_flag = 0
                    sell_flag = 0
                    schedule_mat_2 = np.delete(schedule_mat_2, 0, 0)

        time1 = datetime.datetime.now().astimezone(timezone('America/New_York')).time()

        if ((time1) >= (end_work_time)):
            with open("real result.csv", 'w') as f:
                wr = csv.writer(f)
                wr.writerows(real_result)
            f.close()
            
            print(real_result)
            quit()
