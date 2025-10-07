from pandas import DataFrame
import pandas as pd
from IPython.display import display
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
import re

class EDA:
    def __init__(self,df):
        self.df = df
        self.numeric_columns = self.df.select_dtypes(exclude='object').columns
        self.categorical_columns = self.df.select_dtypes(include='object').columns

    def data_overview(self,df:DataFrame):
        """
        returns the overview of dataset like 
        1. number of rows and columns
        2. Column datatypes
        3. number of rows with missing values
        4. catgorical columns and its unique values
        """
        print(f'shape of the dataset --> rows : {df.shape[0]} and columns : {df.shape[1]}\n')
        print(f"Column types:\n{df.dtypes}\n")
        print(f"number of rows with missing values :\n{df.isnull().sum()}\n")

        resulting_df = {}
        for column in self.categorical_columns:
            resulting_df[column] = pd.Series(df[column].unique())
        
        final_df = pd.DataFrame(resulting_df)
        final_df = final_df.fillna('')
        print(f"dataframe with catgorical columns and its unique values: ")
        display(final_df)
    
    def summary_statistics(self,df:DataFrame):
        """
        parameters: pandas dataframe
        returns Descriptive stats of numeric and categorical columns
        """
        print('Descriptive stats for numerical columns:')
        display(df[self.numeric_columns].describe())
        print()
        print('Descriptive stats for categorical columns:')
        display(df[self.categorical_columns].describe())

    def data_distribution_analysis(self,df:DataFrame):
        """
        parameters: pandas dataframe
        returns Histograms, box plots, skewness, outliers analysis
        """
        n = len(list(df.select_dtypes(exclude='object').columns))
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        axes = axes.flatten()
        print('Histograms of numerical columns: ')
        for i, col in enumerate(df.select_dtypes(exclude='object').columns):
            sns.histplot(df[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        plt.clf()

        print('Boxplots of numerical columns: ')
        for i, col in enumerate(df.select_dtypes(exclude='object').columns):
            sns.boxplot(df[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
            # axes[i].set_xlabel(col)
            # axes[i].set_ylabel('Count')

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        print('Skewness of numerical columns: ')
        for col in self.numeric_columns:
            print(f"skewness of column {col} : {df[col].skew()}")
        print()
        
        print('Outliers per column (using IQR method):')
        for col in self.numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            print(f"Outliers in column {col}:")
            print(outliers.tolist())  # List of outlier values
            print(f"Number of outliers: {len(outliers)}\n")
        
        

class DegreeTransformation(BaseEstimator,TransformerMixin):
    def __init__(self,column):
        self.column = column
        self.degree_map = {
            'some high school': 'high_school',
            'high school': 'high_school',
            'secondary school': 'high_school',
            'ssc': 'high_school',
            'hsc': 'high_school',
            '10th': 'high_school',
            '12th': 'high_school',
            'some college' : 'high_school',
            "associate's degree" : "associate", }
        
    def normalize_degree(self, val):
        val_lower = str(val).lower()
        if re.search(r'master|m\.sc|mtech|m\.tech|msc|ms|post\-graduate', val_lower):
            return 'masters'
        if re.search(r'bachelor|b\.sc|btech|b\.tech|bsc|bs', val_lower):
            return 'bachelors'
        if re.search(r'phd|ph\.d|doctorate', val_lower):
            return 'phd'
        if val_lower in self.degree_map:
            return self.degree_map[val_lower]
        return val
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        X_new = X.copy()
        X_new[self.column] = X_new[self.column].apply(self.normalize_degree)
        return X_new
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [self.column]
        return [f"normalized_{col}" for col in input_features]
        










        

    
