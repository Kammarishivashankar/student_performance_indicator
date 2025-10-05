from pandas import DataFrame
import pandas as pd
from IPython.display import display

class EDA:
    def __init__(self,df):
        self.df = df

    def data_overview(self,df:DataFrame):
        """
        returns the overview of dataset like 
        1. number of rows and columns
        2. Column datatypes
        3. number of rows with missing values
        4. 
        """
        print(f'shape of the dataset --> rows : {df.shape[0]} and columns : {df.shape[1]}\n')
        print(f"Column types:\n{df.dtypes}\n")
        print(f"number of rows with missing values :\n{df.isnull().sum()}\n")

        numeric_columns = df.select_dtypes(exclude='object').columns
        categorical_columns = df.select_dtypes(include='object').columns

        resulting_df = {}
        for column in categorical_columns:
            resulting_df[column] = pd.Series(df[column].unique())
        
        final_df = pd.DataFrame(resulting_df)
        final_df = final_df.fillna('')
        print(f"dataframe with catgorical columns and its unique values: ")
        display(final_df)






        

    
