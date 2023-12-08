import metaflow 
from metaflow import FlowSpec, step, Parameter, IncludeFile, current, JSONType
from datetime import datetime
import os
from io import StringIO
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
import pandas as pd
from comet_ml import Experiment
from comet_ml.integration.metaflow import comet_flow
# assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
# assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
from sklearn.model_selection import ParameterGrid


class LoanFlow(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file
    and training a model successfully.
    """
    
    
    X_testing = IncludeFile(
        'X_testing',
        default='X_test.csv',
        encoding='latin-1'
    )

    y_testing = IncludeFile(
        'y_testing',
        default='y_test.csv',
        encoding='latin-1'
    )
    
    X_training = IncludeFile(
        'X_training',
        default='X_train.csv',
        encoding='latin-1'
    )

    y_training = IncludeFile(
        'y_training',
        default='y_train.csv',
        encoding='latin-1'
    )
        
    # hyperparams = {
    #         "n_estimators": [100,200,300,500,1000],
    #         "criterion": ["gini","entropy","log_loss"],
    #         "max_depth": [None,3,6],
    #         "max_features": [None,"sqrt","log2"],
    # }

    hyperparams = {
            "n_estimators": [50,100,200,300,500,1000],
            "criterion": ["gini","entropy","log_loss"],
            "max_depth": [None,3,6],
            "max_features": [None,"sqrt","log2"],
            "min_samples_split": [1,2,5,10],
            "min_samples_leaf": [1,2,3,4]   
        }

    # hyperparams = {
    #         "n_estimators": [100,200],
    #         "criterion": ["gini","entropy","log_loss"],
    #         "max_depth": [3,6],
    #         "max_features": ["sqrt","log2"],
    # }

    param_grid = list(ParameterGrid(hyperparams))

    hyperparameters = Parameter('hyperparameters',
                      help='list of min_example values',
                      type=JSONType,
                      default=param_grid)

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
        self.X_train = pd.read_csv(StringIO(self.X_training))
        self.y_train = pd.read_csv(StringIO(self.y_training))
        self.next(self.validation_split)

    
    @step
    def validation_split(self):
        from sklearn.model_selection import train_test_split
        #self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train_data, self.y_train_data, test_size=0.1, random_state=42)
        self.X_subset, _, self.y_subset, _ = train_test_split(self.X_train, self.y_train, train_size=0.2, random_state=42)

        # Step 2: Set aside a validation set
        self.X_subset, self.X_valid, self.y_subset, self.y_valid = train_test_split(self.X_subset, self.y_subset, test_size=0.2, random_state=42)


        self.next(self.foreach)


    @step
    def foreach(self):
        self.next(self.validate_model, foreach='hyperparameters')

    
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

    
    @step
    def model_selection(self, inputs):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        self.best_score = 0
        self.best_params = None
        self.best_model = None
    
        experiment = Experiment(
          api_key="wTi6lnFLwxEXgt3I27nbS0PU3",
          project_name="final-project",
          workspace="noremac"
        )
        
        for input in inputs:            
            y_pred = input.model.predict(input.X_valid)  
            y_pred_proba = input.model.predict_proba(input.X_valid)[:,1]
            self.score = roc_auc_score(input.y_valid, y_pred_proba)
            
            params = {
                "n_estimators": input.params[3],
                "criterion": input.params[0],
                "max_depth": input.params[1],
                "max_features": input.params[2],  
            }
            
            experiment.log_parameters(params)
            experiment.log_metric("ROC AUC Score", self.score)
            
            if self.score > self.best_score:
                self.best_score = self.score
                self.best_params = input.params
                self.best_model = input.model
            
        
        # self.next(self.train_model)
        self.next(self.predict)

    
    # @step
    # def train_model(self):
        
        
    #     self.next(self.predict)
        

    @step
    def predict(self):
        from sklearn.metrics import roc_auc_score, classification_report
        from sklearn.metrics import RocCurveDisplay
        from sklearn.ensemble import RandomForestClassifier

        experiment = Experiment(
          api_key="TJq3FJTapE0fH1ke6liHFpEZa",
          project_name="final-project",
          workspace="noremac19"
        )
        
        self.X_test = pd.read_csv(StringIO(self.X_testing))
        self.y_test = pd.read_csv(StringIO(self.y_testing))
        self.X_train = pd.read_csv(StringIO(self.X_training))
        self.y_train = pd.read_csv(StringIO(self.y_training))
        
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

        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:,1]
        
        print(classification_report(self.y_test,y_pred))
        
        score = roc_auc_score(self.y_test, y_pred_proba)
        print("ROC AUC Score: ", score)
        
        experiment.log_parameters(params)
        experiment.log_metric("ROC AUC Score", self.score)
            
        experiment.log_confusion_matrix(y_true=self.y_test, y_predicted=y_pred,
            title="Confusion Matrix", row_label="Actual Category",
            column_label="Predicted Category")

        curve = RocCurveDisplay.from_predictions(self.y_test, y_pred_proba) 
        
        self.next(self.save_model)

    @step
    def save_model(self):
        import pickle
        filename = 'setsnmodels/best_model.pkl'
        # Serialize the model to a file
        with open(filename, 'wb') as file:
            pickle.dump(self.best_model, file)

        self.next(self.end)
        
    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!".format(datetime.utcnow()))



if __name__ == '__main__':
    LoanFlow()
    



