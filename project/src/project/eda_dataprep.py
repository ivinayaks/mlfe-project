import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
from numpy import asarray
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
import metaflow 
from metaflow import FlowSpec, step, Parameter, IncludeFile, current, JSONType
from datetime import datetime
import os
from io import StringIO
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
# from comet_ml import Experiment


class LoanFlow(FlowSpec):

    DATA = IncludeFile(
        'DATA',
        default='loans50k.csv',
        encoding='latin-1'
    )

    @step
    def start(self):
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self):
        df=pd.read_csv(StringIO(self.DATA))
        pd.set_option('display.max_columns', None)
        df.drop(columns=['totalPaid','loanID'],axis=1,inplace=True)
        print(df.head())
        print("Dimensions of the dataset:",df.shape)
        print(df.describe())
        self.df=df
        self.next(self.eda_dataprep)

    @step
    def eda_dataprep(self):
        for col in self.df.columns:
            print("Column Name:",col,"| Column type:",self.df[col].dtype)
            print('-----------------------------------')
        col_object=[]
        for col in self.df.columns:
            if(self.df[col].dtype not in (np.dtype("int64"), np.dtype("float64"))):
                col_object.append(col)
        print("Columns of type object are:",col_object)   
        self.col_object=col_object

        print("COL:term")
        print('-------------')
        print("Number of NaNs:",self.df['term'].isnull().sum())
        self.df.dropna(subset=['term'], inplace=True)
        term_counts=self.df['term'].value_counts()
        print(term_counts)
        plt.title("COL:term")
        plt.bar(term_counts.index,term_counts.values)
        plt.show()
        le=LabelEncoder()
        self.df['term']=le.fit_transform(self.df['term'])
        term_counts=self.df['term'].value_counts()
        print(term_counts)

        print("COL:grade")
        print('-------------')
        print("Number of NaNs:",self.df['grade'].isnull().sum())
        grade_counts=self.df['grade'].value_counts()
        print(grade_counts)
        plt.title("COL:grade")
        plt.bar(grade_counts.index,grade_counts.values)
        plt.show()
        self.df=pd.get_dummies(self.df,columns=['grade'])

        print("COL:employment")
        print('-------------')
        print("Number of NaNs:",self.df['employment'].isnull().sum())
        print('----------------------')
        emp_counts=self.df['employment'].value_counts()
        print(emp_counts)
        print('----------------------')
        print("Unique Values:",len(self.df['employment'].unique()))
        self.df.drop(columns=['employment'],axis=1,inplace=True)

        print("COL:length")
        print('-------------')
        print("Number of NaNs:",self.df['length'].isnull().sum())
        self.df.dropna(subset=['length'], inplace=True)
        length_counts=self.df['length'].value_counts()
        print(length_counts)
        print(length_counts.index)
        oe= OrdinalEncoder(categories=[[None,'< 1 year','1 year','2 years','3 years','4 years','5 years',
                                    '6 years','7 years','8 years','9 years','10+ years']])
        oe.fit(asarray(self.df['length']).reshape(-1,1))
        self.df['length'] = oe.transform(asarray(self.df['length']).reshape(-1,1))
        length_counts=self.df['length'].value_counts()
        plt.title("COL:length")
        plt.bar(length_counts.index,length_counts.values)
        plt.figure(figsize=(10,6))
        plt.show()

        print("COL:home")
        print('-------------')
        print("Number of NaNs:",self.df['home'].isnull().sum())
        home_counts=self.df['home'].value_counts()
        print(home_counts)
        plt.title("COL:home")
        plt.bar(home_counts.index,home_counts.values)
        plt.show()
        self.df=pd.get_dummies(self.df,columns=['home'])

        print("COL:verified")
        print('-------------')
        print("Number of NaNs:",self.df['verified'].isnull().sum())
        verified_counts=self.df['verified'].value_counts()
        print(verified_counts)
        plt.title("COL:verified")
        plt.bar(verified_counts.index,verified_counts.values)
        plt.show()
        self.df=pd.get_dummies(self.df,columns=['verified'])

        print("Number of NaNs:",self.df['status'].isnull().sum())
        counts=self.df['status'].value_counts()
        print(counts)
        self.df = self.df[~self.df['status'].isin(['Current','Late (31-120 days)','In Grace Period',
                                    'Late (16-30 days)'])]
        
        counts=self.df['status'].value_counts()
        print(counts)
        plt.title("COL:status")
        plt.bar(counts.index,counts.values)
        plt.show()
        self.df['status']=[1 if val in ['Charged Off','Default'] else 0 for val in self.df['status']]
        counts=self.df['status'].value_counts()
        print(counts)

        print("COL:reason")
        print('-------------')
        print("Number of NaNs:",self.df['reason'].isnull().sum())
        counts=self.df['reason'].value_counts()
        print(counts)
        self.df['reason']=['debt_consolidation' if val=='debt_consolidation' else 'credit_card' if val=='credit_card'
             else 'other' for val in self.df['reason']]
        counts=self.df['reason'].value_counts()
        print(counts)
        plt.title("COL:reason")
        plt.bar(counts.index,counts.values)
        plt.show()
        self.df=pd.get_dummies(self.df,columns=['reason'])

        print("COL:state")
        print('-------------')
        print("Number of NaNs:",self.df['state'].isnull().sum())
        counts=self.df['state'].value_counts()
        print(counts)
        self.df=pd.get_dummies(self.df,columns=['state'])

        num_columns=[]
        cat_columns=[]
        for col in self.df.columns:
            if(col=='status'):
                continue
            if(self.df[col].dtype in (np.dtype("int64"), np.dtype("float64"))):
                num_columns.append(col)
            else:
                cat_columns.append(col)

        nan_columns=[]
        for col in num_columns:
            if(self.df[col].isnull().sum()>0):
                nan_columns.append(col)

        for col in nan_columns:
            print(self.df[col].describe())

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        for col in nan_columns:
            self.df[col]=imputer.fit_transform(self.df[col].values.reshape(-1,1))[:,0]

        # fig , axes = plt.subplots(nrows=6, ncols=4,constrained_layout=True)       
        # fig.subplots_adjust(left= 0, bottom=0, right=3, top=12, wspace=0.09, hspace=0.3)
        # for ax, column in zip(axes.flatten(),num_columns):
        #     sns.boxplot(self.df[column],ax=ax)
        # plt.show()

        def plot_boxplots(df,colums):
            fig , axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)       
            fig.subplots_adjust(left= 0, bottom=0, right=3, top=6, wspace=0.04, hspace=0.1)
            for ax, column in zip(axes.flatten(),colums):
                sns.boxplot(df[column],ax=ax)
        
        plot_boxplots(self.df,num_columns[:6])
        plot_boxplots(self.df,num_columns[6:12])
        plot_boxplots(self.df,num_columns[12:18])
        plot_boxplots(self.df,num_columns[18:])

        def plot_numerical_distributions(df, columns, target_column):
            fig = plt.figure(figsize=(10, 15))
            for i, column in enumerate(columns):
                plt.subplot(2, 3, i + 1)
                sns.distplot(df[df[target_column] == 1][column], hist=False, label="1")
                sns.distplot(df[df[target_column] == 0][column], hist=False, label="0")
                plt.legend()
            plt.show()

        plot_numerical_distributions(self.df,num_columns[:6],'status')
        plot_numerical_distributions(self.df,num_columns[6:12],'status')
        plot_numerical_distributions(self.df,num_columns[12:18],'status')
        plot_numerical_distributions(self.df,num_columns[18:],'status')



        self.df_=self.df.copy()
        self.df_.drop(columns=cat_columns,axis=1,inplace=True)
       
        plt.figure(figsize=(10,10))
        plt.title("Heatmap of correlations")
        sns.heatmap(self.df_.corr())

        corrs=self.df.corr()['status']
        corrs_index=corrs.index
        corrs_values=corrs.values
        imp_cols=[]
        for i in range(len(corrs)):
            if(corrs_index[i]=='status'): 
                continue
            if(corrs_index[i] in num_columns and corrs_values[i]>0.1):
                imp_cols.append(corrs_index[i])
        print("Possibly pivotal columns:",imp_cols)

        labels=['defaults','no defaults']
        show=[self.df['status'].value_counts().values[1]/self.df['status'].value_counts().values.sum(),
            self.df['status'].value_counts().values[0]/self.df['status'].value_counts().values.sum()]
        fig1, ax1 = plt.subplots()
        ax1.pie(show,labels=labels,startangle=110)
        ax1.axis('equal')
        plt.title('Data imbalance',fontsize=25)
        plt.show()
        print("Percentage of defaults:",self.df['status'].value_counts().values[1]/self.df['status'].value_counts().values.sum())
        print("Percentage of non-defaults:",self.df['status'].value_counts().values[0]/self.df['status'].value_counts().values.sum())

        y=self.df['status']
        X=self.df.copy()
        X.drop(columns=['status'],axis=1,inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=26)
        sm = SMOTE()
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)

        self.next(self.end)
    
    @step
    def end(self):
        print("All done")





        











if __name__ == '__main__':
    LoanFlow()