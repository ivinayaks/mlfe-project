from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from numpy import asarray
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# ... (include other necessary imports)

app = Flask(__name__)

# Load the model
model_filename = 'setsnmodels/best_model.pkl'  # Replace with your model's path
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def preprocess_data(df):

        le=LabelEncoder()
        df['term']=le.fit_transform(df['term'])

        df=pd.get_dummies(df,columns=['grade'])

        oe= OrdinalEncoder(categories=[[None,'< 1 year','1 year','2 years','3 years','4 years','5 years',
                                    '6 years','7 years','8 years','9 years','10+ years']])
        oe.fit(asarray(df['length']).reshape(-1,1))
        df['length'] = oe.transform(asarray(df['length']).reshape(-1,1))

        df=pd.get_dummies(df,columns=['home'])

        df=pd.get_dummies(df,columns=['verified'])

        df['reason']=['debt_consolidation' if val=='debt_consolidation' else 'credit_card' if val=='credit_card'
             else 'other' for val in df['reason']]

        df=pd.get_dummies(df,columns=['reason'])

        df=pd.get_dummies(df,columns=['state'])

        return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = preprocess_data(df)
        prediction = model.predict(df)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
