import numpy as np
import pandas as pd
from datetime import datetime


dataset = "Police_Department_Incident_Reports__2018_to_Present.csv"


def read_data(file_name):
    dataframe = pd.read_csv(file_name)
    dataframe.drop(columns=['Row ID', 'Incident ID', 'Incident Number',
                            'CAD Number', 'Report Type Description',
                            'Incident Code', 'Incident Subcategory', 'Incident Description',
                            'Intersection', 'CNN', 'Neighborhoods', 'ESNCAG - Boundary File',
                            'Central Market/Tenderloin Boundary Polygon - Updated',
                            'Civic Center Harm Reduction Project Boundary',
                            'HSOC Zones as of 2018-06-05', 'Invest In Neighborhoods (IIN) Areas',
                            'Current Supervisor Districts', 'Current Police Districts'], axis=1, inplace=True)
    print(dataframe.columns)
    dataframe = dataframe.fillna(0)
    #   dataframe['Filed Online'] = dataframe['Filed Online'].fillna(False) # does not work???
    #   print(dataframe['Filed Online'].head(8))
    incident_report_time_difference = []
    myformat = '%Y/%m/%d %I:%M:%S %p'
    for inc, rep in zip(dataframe['Incident Datetime'], dataframe['Report Datetime']):
        if inc != 0 and rep != 0:
            inc_obj = datetime.strptime(inc, myformat)
            rep_obj = datetime.strptime(rep, myformat)
            tmp = rep_obj - inc_obj
            incident_report_time_difference.append(tmp)
    dataframe['Time Difference'] = incident_report_time_difference
    #   should i remove the rows, where time difference is 0?
    #   clearly the data is not correct, report and incident times
    #   should differ at least a little



def main():
    read_data(dataset)
    pass


if __name__ == '__main__':
    main()
