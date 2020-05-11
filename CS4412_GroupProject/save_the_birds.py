import pandas as pd

"""
Authors: Levi Sutton, Hope Shackelford, Rico Santiago
Objective: To sort the data set, Bird Strikes Test.csv, using Apriori algorithm to sort by seasons. 
"""

def cleaning_data():
    # Reads in file as data, which is a data frame type.
    data = pd.read_csv("..\\CS4412_GroupProject\\Bird Strikes Test.csv",
                       usecols=['Airport: Name', 'FlightDate', 'Record ID', 'When: Phase of flight',
                                'Wildlife: Size',
                                'Wildlife: Species'])
    # TODO: Remove null values in the columns. Hope is working on this atm.
    data = data.dropna(how='any', axis=0)
    data = data[~data['FlightDate'].isnull()]
    # Converts data to string so it's easy to print.
    df = data.to_string()
    print(df)
    return df

def regression():
    # Gets cleaned data from this method.
    df = cleaning_data()

    # https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/
    # airport name
    # item phase of flight

    # date trans categorize by season
    # items airport name

    # key is airport name primary key group them together by apriori
    # categorize further

    # we force to have one name and one phase

"""
This is the main method where it starts the program. 
"""
def main():
    cleaning_data()
    #apriori()

main() # This line starts the whole program.