# Project on Credit Default Prediction built for Machine Learning and Financial Engineering course at NYU
Credit Default Prediction Model 

In this project, we created a model used to predict mortgage loan default. 

The dataset that we used was acquired from the following website: https://datascienceuwl.github.io/Project2018/loans50k.csv. 
The dataset is comprised of 50,000 lines of 50,000 loans with the following variables: loanID,amount,term,rate,payment,grade,employment,length,home,income,verified,status,reason,state,debtIncRat,delinq2yr,inq6mth,openAcc,pubRec,revolRatio,totalAcc,totalPaid,totalBal,totalRevLim,accOpen24,avgBal,bcOpen,bcRatio,totalLim,totalRevBal,totalBcLim,totalIlLim. 

The predicted label is status. 

More information on the description of the dataset can be found at: https://datascienceuwl.github.io/Project2018/TheData.html . 

We performed some data preprocessing and exploratory data analysis to determine the features for the model. We trained and tested the various models like Logistic Regression, Random Forest, Decision Trees, and Support Vector Machines. Then, we completed hyperparameter tuning for a few hyperparameters in the RandomForestClassifier model: n_estimators, criterion, max_depth, and max_features. We used comet to log the results of the hyperparameter tuning and model testing. The results of the logging can be found at the following url: https://www.comet.com/noremac/final-project?shareable=sKyvqChRvFhW679WAx5L9vpRG 

Qualitative & Robustness Checks: We performed a data slice analysis on a categorical feature (‘Grade’). There are six different values for grade: [‘A’, ‘B’, ‘C’, ‘D’, ‘E’, ‘F’, ‘G’].  We ran a robustness check with noise added to a numerical feature (‘Rate’). 

Model Accuracy: To evaluate model accuracy, we computed the ROC AUC Score, Accuracy, Precision, Recall, F1-Score, ROC Curves, and Confusion Matrices. 

Instructions to run the model:
1. Use the following command to run the model: COMET_API_KEY='Insert API Key' python loanv3.py run --max-num-splits=150
    The loanv3.py runs the LoanFLow metaflow. Within the metaflow, there are steps related to EDA, data preprocessing, hyperparameter tuning, and testing. 
2. Run the Flask back-end in one terminal: python flaskapp.py
3. Use the following command to run the streamlit app in another terminal: streamlit run streamlit.py
4. Play around with the inputs on the Streamlit app. Loans with grade 'A' are likely to not default, while loan with grade 'G' are likely to default.
![image](https://github.com/ivinayaks/mlfe-project/assets/54876656/00cb736e-3c23-4be1-a26d-5391e962ac19)
