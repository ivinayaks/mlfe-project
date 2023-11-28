import metaflow 
from metaflow import FlowSpec, step, Parameter, IncludeFile, current, JSONType
from datetime import datetime
import os
from io import StringIO
import pylint
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
import pandas as pd
# from comet_ml import Experiment
# from comet_ml.integration.metaflow import comet_flow
# assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
# assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']



class LoanFlow(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file
    and training a model successfully.
    """
    
    
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
        """
        Read the data in from the static file
        """
        self.raw_data = pd.read_csv(StringIO(self.DATA))

        print(self.raw_data)
        self.next(self.preprocess)


    @step
    def preprocess(self):
        df = self.raw_data
        status_mapping = {
            'Fully Paid': 0,
            'Current': 0,
            'Charged Off': 1,
            'In Grace Period': 1,
            'Late (16-30 days)': 1,
            'Late (31-120 days)': 1,
            'Default': 1
        }
        
        df['status'] = df['status'].map(status_mapping).fillna(0).astype(int)

        df.dropna(inplace=True)
        self.preprocessed_data = df
        self.next(self.prepare_dataset)
        
    @step
    def prepare_dataset(self):
        from sklearn.model_selection import train_test_split
        df = self.preprocessed_data
        
        self.num_columns = list(df.select_dtypes(include='number').columns)
        self.cat_columns = list(df.select_dtypes(include='object').columns) 
        
        y = df['status']
        df.drop(columns=['status'],axis=1,inplace=True)
        X = df.copy()

        X = pd.get_dummies(X,columns=self.cat_columns,sparse=True)

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=0.2)
        
        self.next(self.logreg, self.randforest)
        
    @step
    def logreg(self):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        self.model = model.fit(self.X_train,self.y_train)
        self.next(self.predict)
    
    @step
    def randforest(self):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        self.model = model.fit(self.X_train, self.y_train)
        self.next(self.predict)

    @step
    def predict(self, inputs):
        from sklearn.metrics import roc_auc_score
        y_pred = self.model.predict_proba(X_test)[:,1]
        roc_auc_score(y_test,y_pred)
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!".format(datetime.utcnow()))



if __name__ == '__main__':
    LoanFlow()
    



