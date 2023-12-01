import metaflow 
from metaflow import FlowSpec, step, Parameter, IncludeFile, current, JSONType
from datetime import datetime
import os
from io import StringIO
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
        import pickle
        # with open('X_train.pkl', 'rb') as file:
        #         self. pickle.load(X_train, file)
        
        with open('setsnmodels/X_test.pkl', 'rb') as file:
                self.X_test = pickle.load(file)

        # with open('y_train.pkl', 'rb') as file:
        #         pickle.load(y_train, file)

        with open('setsnmodels/y_test.pkl', 'rb') as file:
                self.y_test = pickle.load(file)
        
        self.next(self.load_model)
        
    @step
    def load_model(self):
        import pickle
        with open('setsnmodels/best_model.pkl', 'rb') as file:
            self.model = pickle.load(file)
        
        self.next(self.predict)
        
    @step
    def predict(self):
        from sklearn.metrics import roc_auc_score, classification_report

        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test,y_pred))
        score = roc_auc_score(self.y_test,self.model.predict_proba(self.X_test)[:,1])
        print("ROC AUC Score: ", score)

        self.next(self.end)
        
    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!".format(datetime.utcnow()))



if __name__ == '__main__':
    LoanFlow()
    



