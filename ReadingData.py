import pandas as pd
from datetime import datetime


dataset = "Police_Department_Incident_Reports__2018_to_Present.csv"


def read_data(file_name):
    dataframe = pd.read_csv(file_name)

    dataframe = dataframe.head(5000)

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
                            'Report Datetime', 'Row ID', 'Incident ID', 'Incident Number',
                            'CAD Number', 'Report Type Description', 'Incident Code', 'Supervisor District',
                            'Incident Subcategory', 'Incident Description', 'Intersection', 'CNN', 'Neighborhoods',
                            'ESNCAG - Boundary File', 'Central Market/Tenderloin Boundary Polygon - Updated',
                            'Civic Center Harm Reduction Project Boundary', 'HSOC Zones as of 2018-06-05',
                            'Invest In Neighborhoods (IIN) Areas', 'Current Supervisor Districts',
                            'Current Police Districts', 'Latitude', 'Longitude', 'Point'], axis=1, inplace=True)

    dataframe.columns = dataframe.columns.str.replace(' ', '')

    dataframe = dataframe.fillna(0)
    dataframe['FiledOnline'] = dataframe['FiledOnline'].replace(0, False)

    print(dataframe.columns)
    dataframe = dataframe[dataframe.TimeDifference != 0]
    dataframe = dataframe[dataframe.AnalysisNeighborhood != 0]
    dataframe = dataframe[dataframe.IncidentCategory != 0]

    X = dataframe
    Y = X['TimeDifference']
    X = X.drop(['TimeDifference'], axis=1)

    X = pd.get_dummies(
        dataframe, columns=['PoliceDistrict', 'ReportTypeCode', 'IncidentCategory',
                            'AnalysisNeighborhood', 'Resolution', 'IncidentDayofWeek'])

    return X, Y
