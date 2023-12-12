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

encoders_path='setsnmodels/pickle_path.pkl'
with open(encoders_path, 'rb') as file:
     encoders=pickle.load(file)

def preprocess_data(df):

        le_term=encoders[0]
        df['term']=le_term.transform(df['term'])

        oe_grade=encoders[1]
        onehot = oe_grade.transform(df[['grade']])
        feature_names = oe_grade.categories_[0]
        onehot_df = pd.DataFrame(onehot.toarray(), columns=feature_names)
        df.reset_index(drop=True, inplace=True)
        onehot_df.reset_index(drop=True, inplace=True)
        df= pd.concat([df, onehot_df], axis=1)
        df.drop(columns=['grade'], axis=1, inplace=True)

        oe_length=encoders[2]
        df['length'] = oe_length.transform(asarray(df['length']).reshape(-1,1))

        oe_home=encoders[3]
        onehot = oe_home.transform(df[['home']])
        feature_names = oe_home.categories_[0]
        onehot_df = pd.DataFrame(onehot.toarray(), columns=feature_names)
        df.reset_index(drop=True, inplace=True)
        onehot_df.reset_index(drop=True, inplace=True)
        df= pd.concat([df, onehot_df], axis=1)
        df.drop(columns=['home'], axis=1, inplace=True)

        oe_verified=encoders[4]
        onehot = oe_verified.transform(df[['verified']])
        feature_names = oe_verified.categories_[0]
        onehot_df = pd.DataFrame(onehot.toarray(), columns=feature_names)
        df.reset_index(drop=True, inplace=True)
        onehot_df.reset_index(drop=True, inplace=True)
        df= pd.concat([df, onehot_df], axis=1)
        df.drop(columns=['verified'], axis=1, inplace=True)

        df['reason']=['debt_consolidation' if val=='debt_consolidation' else 'credit_card' if val=='credit_card'
             else 'other' for val in df['reason']]
        oe_reason=encoders[5]
        onehot = oe_reason.transform(df[['reason']])
        feature_names = oe_reason.categories_[0]
        onehot_df = pd.DataFrame(onehot.toarray(), columns=feature_names)
        df.reset_index(drop=True, inplace=True)
        onehot_df.reset_index(drop=True, inplace=True)
        df= pd.concat([df, onehot_df], axis=1)
        df.drop(columns=['reason'], axis=1, inplace=True)

        oe_state=encoders[6]
        onehot = oe_state.transform(df[['state']])
        feature_names = oe_state.categories_[0]
        onehot_df = pd.DataFrame(onehot.toarray(), columns=feature_names)
        df.reset_index(drop=True, inplace=True)
        onehot_df.reset_index(drop=True, inplace=True)
        df= pd.concat([df, onehot_df], axis=1)
        df.drop(columns=['state'], axis=1, inplace=True)

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
