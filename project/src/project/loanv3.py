# Import necessary libraries
import metaflow 
from metaflow import FlowSpec, step, Parameter, IncludeFile, current, JSONType
from datetime import datetime
import os
from io import StringIO
import pandas as pd
import numpy as np
from comet_ml import Experiment
from comet_ml.integration.metaflow import comet_flow
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
from numpy import asarray
from imblearn.over_sampling import SMOTE
#warnings.filterwarnings("ignore")
# Check and assert necessary environment variables
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']



# Class definition for the Metaflow flow
class LoanFlow(FlowSpec):
    """
    LoanFlow is a DAG showcasing reading data from files
    and training, validating, and testing a model successfully.
    """

    DATA = IncludeFile(
        'DATA',
        default='loans50k.csv',
        encoding='latin-1'
    )
    
    # Hyperparameter grid for RandomForestClassifier
    hyperparams = {
        "n_estimators": [100],
        "criterion": ["gini","entropy","log_loss"],
        "max_depth": [None,3,6],
        "max_features": [None,"sqrt","log2"],
    }

    param_grid = list(ParameterGrid(hyperparams))

    # Parameter for hyperparameters with default as the hyperparameter grid
    hyperparameters = Parameter('hyperparameters',
                      help='list of min_example values',
                      type=JSONType,
                      default=param_grid)

    # Step 1: Starting point of the flow
    @step
    def start(self):
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    #Step 2: Load data from CSV files
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
        import pickle 
        
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

        # serialize datasets
        with open('setsnmodels/X_train.pkl', 'wb') as file:
            pickle.dump(self.X_train, file)

        with open('setsnmodels/y_train.pkl', 'wb') as file:
            pickle.dump(self.y_train, file)

        with open('setsnmodels/X_test.pkl', 'wb') as file:
            pickle.dump(self.X_test, file)

        with open('setsnmodels/y_test.pkl', 'wb') as file:
            pickle.dump(self.y_test, file)

        self.next(self.validation_split)

    # Step 3: Split data into training subset and validation set
    @step
    def validation_split(self):
        from sklearn.model_selection import train_test_split
        self.X_subset, _, self.y_subset, _ = train_test_split(self.X_train, self.y_train, train_size=0.2, random_state=42)
        self.X_subset, self.X_valid, self.y_subset, self.y_valid = train_test_split(self.X_subset, self.y_subset, test_size=0.2, random_state=42)
        self.next(self.foreach)

    # Step 4: Iterate over hyperparameter grid
    @step
    def foreach(self):
        self.next(self.validate_model, foreach='hyperparameters')

    # Step 5: Train and validate the model with specified hyperparameters
    @step
    def validate_model(self):
        """
        Train a regression on the training set
        """
        from sklearn.ensemble import RandomForestClassifier
        
        self.params = list(self.input.values())
        
        rf = RandomForestClassifier(
            n_estimators = self.params[3],
            criterion = self.params[0],
            max_depth = self.params[1],
            max_features = self.params[2],   
        )
        
        rf.fit(self.X_subset, self.y_subset)
        
        self.model = rf
        
        self.next(self.model_selection)

    # Step 6: Select the best model based on validation performance
    @step
    def model_selection(self, inputs):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        self.best_score = 0
        self.best_params = None
        self.best_model = None
        
        def calculate_additional_metrics(y_true, y_pred, y_pred_proba):
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            return precision, recall, f1, roc_auc

        experiment = Experiment(
            api_key=os.environ['COMET_API_KEY'],
            project_name="final-project",
            workspace="noremac"
        )

        for input in inputs:
            y_pred = input.model.predict(input.X_valid)  
            y_pred_proba = input.model.predict_proba(input.X_valid)[:, 1]
            precision, recall, f1, roc_auc = calculate_additional_metrics(input.y_valid, y_pred, y_pred_proba)

            params = {
                "n_estimators": input.params[3],
                "criterion": input.params[0],
                "max_depth": input.params[1],
                "max_features": input.params[2],  
            }
            
            # Log metrics using Comet ML
            experiment.log_metric("Precision", precision)
            experiment.log_metric("Recall", recall)
            experiment.log_metric("F1-Score", f1)
            experiment.log_metric("ROC AUC Score", roc_auc)
            
            # Check and update the best model
            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_params = input.params
                self.best_model = input.model
            
        self.next(self.predict)



    # Step 7: Predict using the best model on the test set
    @step
    def predict(self):
        from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
        from sklearn.metrics import RocCurveDisplay
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        import pickle
        
        experiment = Experiment(
        api_key=os.environ['COMET_API_KEY'],
        project_name="final-project",
        workspace="noremac"
        )
        
        # Load the datasets
        with open('setsnmodels/X_train.pkl', 'rb') as file:
            self.X_train = pickle.load(file)
        with open('setsnmodels/y_train.pkl', 'rb') as file:
            self.y_train = pickle.load(file)
        with open('setsnmodels/X_test.pkl', 'rb') as file:
            self.X_test = pickle.load(file)
        with open('setsnmodels/y_test.pkl', 'rb') as file:
            self.y_test = pickle.load(file)
        
        # Create and train the best model
        model = RandomForestClassifier(
            n_estimators = self.best_params[3],
            criterion = self.best_params[0],
            max_depth = self.best_params[1],
            max_features = self.best_params[2],   
        )
        
        params = {
                "n_estimators": self.best_params[3],
                "criterion": self.best_params[0],
                "max_depth": self.best_params[1],
                "max_features": self.best_params[2],  
        }

        
        model.fit(self.X_train, self.y_train)
        self.best_model = model

        # Make predictions and evaluate performance
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:,1]
        
        # Print and log classification report
        report = classification_report(self.y_test, y_pred)
        print(report)
        experiment.log_text("Classification Report", report)
        
        # Calculate ROC AUC Score
        score = roc_auc_score(self.y_test, y_pred_proba)
        print("ROC AUC Score: ", score)
        
        # Log parameters and metrics
        experiment.log_parameters(params)
        experiment.log_metric("ROC AUC Score", score)  # Corrected here
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='g')
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        # Log confusion matrix using Comet ML
        experiment.log_confusion_matrix(y_true=self.y_test, y_predicted=y_pred,
                                        title="Confusion Matrix", row_label="Actual",
                                        column_label="Predicted")
        
        curve = RocCurveDisplay.from_predictions(self.y_test, y_pred_proba) 
        curve.plot()
        plt.show()
        
        self.next(self.behavioral_tests)



    # Step 8: Testing (Qualitative and Slice) and Robustness Check
    @step
    def behavioral_tests(self):
        """
        Perform qualitative checks, data slice performance analysis, and robustness checks.
        """
        # Qualitative Check: Review some random predictions
        self.qualitative_check()

        # Data Slice Analysis: Check model performance on different segments of data
        self.data_slice_analysis()

        # Robustness Check: Evaluate model performance on perturbed data
        self.robustness_check()

        self.next(self.save_model)

    def qualitative_check(self):
        import random
        print("\n--- Qualitative Check ---")
        sample_indices = random.sample(range(len(self.X_test)), 5)
        for i in sample_indices:
            print(f"Test Sample {i}:")
            print(f"Features: {self.X_test.iloc[i]}")
            print(f"Actual Label: {self.y_test.iloc[i]}, Predicted Label: {self.best_model.predict([self.X_test.iloc[i]])}\n")

    def data_slice_analysis(self):
        from sklearn.metrics import accuracy_score, classification_report
        print("\n--- Data Slice Analysis on 'grade' ---")
        # Analyze performance based on a categorical feature like 'grade'
        if 'grade' in self.X_test.columns:
            for grade in self.X_test['grade'].unique():
                mask = self.X_test['grade'] == grade
                print(f"Performance for loan grade {grade}:")
                print(classification_report(self.y_test[mask], self.best_model.predict(self.X_test[mask])))
                print("Accuracy:", accuracy_score(self.y_test[mask], self.best_model.predict(self.X_test[mask])), "\n")

    def robustness_check(self):
        from sklearn.metrics import accuracy_score, classification_report
        print("\n--- Robustness Check on 'rate' ---")
        # Add random noise to a numerical feature like 'rate'
        perturbed_X_test = self.X_test.copy()
        if 'rate' in perturbed_X_test.columns:
            noise = np.random.normal(0, 1, perturbed_X_test['rate'].shape)
            perturbed_X_test['rate'] += noise
            print("Performance on perturbed 'loan_amount' data:")
            print(classification_report(self.y_test, self.best_model.predict(perturbed_X_test)))
            print("Accuracy:", accuracy_score(self.y_test, self.best_model.predict(perturbed_X_test)), "\n")



    # Step 9: Save the best model to a file using pickle
    @step
    def save_model(self):
        import pickle
        filename = 'setsnmodels/best_model.pkl'
        # Serialize the model to a file
        with open(filename, 'wb') as file:
            pickle.dump(self.best_model, file)

        self.next(self.end)
        
    # Step 10: End step, print a completion message
    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!".format(datetime.utcnow()))

# Main block: Execute the Metaflow flow
if __name__ == '__main__':
    LoanFlow()
