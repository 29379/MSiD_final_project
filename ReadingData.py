from collections import Counter

import pandas as pd
from datetime import datetime
import numpy as np


dataset = "Police_Department_Incident_Reports__2018_to_Present.csv"


def read_data(file_name):
    dataframe = pd.read_csv(file_name)

    dataframe = dataframe.sample(n=5000)

    incident_report_time_difference = []
    my_format = '%Y/%m/%d %I:%M:%S %p'
    for inc, rep in zip(dataframe['Incident Datetime'], dataframe['Report Datetime']):
        if inc != 0 and rep != 0:
            inc_obj = datetime.strptime(inc, my_format)
            rep_obj = datetime.strptime(rep, my_format)
            tmp = rep_obj - inc_obj
            minutes_passed = float(tmp.total_seconds() / 60)
            incident_report_time_difference.append(minutes_passed)
    dataframe['TimeDifference'] = incident_report_time_difference

    dataframe.drop(columns=['Incident Datetime', 'Incident Date', 'Incident Time', 'Incident Year',
                            'Report Datetime', 'Row ID', 'Incident ID', 'Incident Number', 'Analysis Neighborhood',
                            'CAD Number', 'Report Type Description', 'Incident Code', 'Supervisor District',
                            'Incident Subcategory', 'Incident Description', 'Intersection', 'CNN', 'Neighborhoods',
                            'ESNCAG - Boundary File', 'Central Market/Tenderloin Boundary Polygon - Updated',
                            'Civic Center Harm Reduction Project Boundary', 'HSOC Zones as of 2018-06-05',
                            'Invest In Neighborhoods (IIN) Areas', 'Current Supervisor Districts',
                            'Current Police Districts', 'Latitude', 'Longitude', 'Point'], axis=1, inplace=True)

    dataframe.columns = dataframe.columns.str.replace(' ', '')

    dataframe = dataframe.fillna(0)
    dataframe['FiledOnline'] = dataframe['FiledOnline'].replace(True, 1)
    dataframe['FiledOnline'] = dataframe['FiledOnline'].replace(False, 0)

    q1 = np.percentile(dataframe['TimeDifference'], 5)
    q3 = np.percentile(dataframe['TimeDifference'], 95)
    iqr = q3 - q1
    limit = 1.5 * iqr

    dataframe = dataframe[dataframe.TimeDifference > 0]
    dataframe = dataframe[dataframe.TimeDifference > (q1 - limit)]
    dataframe = dataframe[dataframe.TimeDifference < (q1 + limit)]
    dataframe = dataframe[dataframe.IncidentCategory != 0]

    print(dataframe.columns)

    X = dataframe
    Y = X['TimeDifference']
    X.drop(['TimeDifference'], axis=1, inplace=True)

    X = pd.get_dummies(
        dataframe, columns=['PoliceDistrict', 'ReportTypeCode', 'IncidentCategory',
                            'Resolution', 'IncidentDayofWeek'])

    return X, Y
