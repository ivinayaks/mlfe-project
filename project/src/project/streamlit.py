import streamlit as st
import requests

# Function to send data to the Flask endpoint and get the prediction
def get_prediction(data):
    url = "http://127.0.0.1:5000/predict"  # Replace with the correct URL if different
    response = requests.post(url, json=data)
    # Check if the response is valid
    if response.status_code == 200:
        prediction = response.json()
        # Debugging: Print the whole response
        #st.write("Full response:", prediction)

        # Check if 'prediction' key exists in the response
        if 'prediction' in prediction:
            st.write(f"Prediction (Will Default?): {prediction['prediction']}")
            if prediction['prediction'] == [1]:
                st.subheader("The model predicts that the loan will default.")
            else:
                st.subheader("The model predicts that the loan will not default.")
            #st.write(type(prediction['prediction']))
            
       
        else:
            st.error("Prediction key not found in response.")
        
    else:
        st.error(f"Failed to get prediction. Status code: {response.status_code}")
    
    return response.json()

# Streamlit UI
st.title("Loan Prediction App")

# Creating input fields for each feature
amount = st.number_input("Loan Amount in USD", min_value=0, value = 8000)
term = st.selectbox("Loan Term", ["36 months", "60 months"], index = 0)
rate = st.number_input("Interest Rate as a decimal", min_value=0.0, max_value=1.0, step=0.01, value = 0.14)
payment = st.number_input("Monthly Payment Amount", min_value=0.0 , value = 272.07
                          )
grade = st.selectbox("Grade of loan", ["A", "B", "C", "D", "E", "F", "G"], index = 2)
length = st.selectbox("Length of Employment", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                                               "6 years", "7 years", "8 years", "9 years", "10+ years"], index = 3)
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"], index = 0)
income = st.number_input("Annual Income", min_value=0, value = 49966)
verified = st.selectbox("Verification Status of Annual Income", ["Not Verified", "Source Verified", "Verified"], index = 1)
reason = st.selectbox("Applicant's Purpose for the Loan", ["credit_card", "debt_consolidation", "other", "home_improvement", "car","major_purchase","medical","moving","vacation","house","renewable_energy"], index = 0)
state = st.selectbox("State Of Residence", ["CA", "PA", "NY", "TX", "FL", "OH", "WA", "VA", "MN", "NC", "RI", "LA", "WI", "NJ", "AK", "AL", "AR", "AZ", "CO", "CT", "DC", "DE", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "MA", "MD", "ME", "MI", "MO", "MS", "MT", "ND", "NE", "NH", "NM", "NV", "OK", "OR", "SC", "SD", "TN", "UT", "VT", "WV", "WY"], index =0)
debtIncRat = st.number_input("Monthly Non-Mortgage Debt Payment to Monthly Income Ratio", min_value=0.0, step=0.01, value = 30.05)
delinq2yr = st.number_input("Number of 30+ day late payments in last two year", min_value=0.0 , max_value=10.0, step=1.0, value=0.0)
inq6mth = st.number_input("Credit Checks in Last 6 Months", min_value=0.0, max_value=10.0, step=1.0, value = 0.0)
openAcc = st.number_input("Number of Open Credit Lines", min_value=0, max_value=100, step = 1, value = 11)
pubRec = st.number_input("Number of Derogatory Public Records such as Bankruptcy", min_value=0.0, max_value=10.0, step=1.0, value = 0.0)
revolRatio = st.number_input("Revolving Line Utilization Rate", min_value=0.0, max_value=1.0, step=0.01, value = 0.55)
totalAcc = st.number_input("Total Number of Credit Lines in File", min_value=0, max_value=1000, step=1, value = 15)
totalBal = st.number_input("Total Current Balance of All Credit Accounts", min_value=0.0 , max_value=10000000.0, step=0.1, value = 48054.0)
totalRevLim = st.number_input("Total Revolving Credit Limit", min_value=0.0 , max_value=10000000.0, step=0.1,value = 8100.0)
accOpen24 = st.number_input("Accounts Opened in Last 24 Months", min_value=0 , max_value=100, step=1, value = 8)
avgBal = st.number_input("Average Account Balance", min_value=0.0 , max_value=10000000.0, step=0.1, value = 4369.0)
bcOpen = st.number_input("Total Unused Credit on Credit Card", min_value=0.0 ,max_value=10000000.0, step=0.1, value = 43.0)
bcRatio = st.number_input("Ratio of total credit card balance to total credit card lmits", min_value=0.0, max_value=100.0, step=0.01, value = 95.7)
totalLim = st.number_input("Total Credit Limit", min_value=0.0 , max_value=10000000.0, step=0.1,value = 60629.0)
totalRevBal = st.number_input("Total Credit Balance Except Mortgages", min_value=0.0 , max_value=10000000.0, step=0.1,value = 48054.0)
totalBcLim = st.number_input("Total Credit Limit of Credit Cards", min_value=0.0 , max_value=10000000.0, step=0.1,value = 1000.0 )
totalIlLim = st.number_input("Total Credit Limit for Installment Accounts", min_value=0.0 ,max_value=10000000.0, step=0.1, value = 52529.0)

# Button to make prediction
if st.button('Predict Loan Status'):
    # Create a dictionary with the input data
    data = {
        "amount": amount,
        "term": term,
        "rate": rate,
        "payment": payment,
        "grade": grade,
        "length": length,
        "home": home,
        "income": income,
        "verified": verified,
        "reason": reason,
        "state": state,
        "debtIncRat": debtIncRat,
        "delinq2yr": delinq2yr,
        "inq6mth": inq6mth,
        "openAcc": openAcc,
        "pubRec": pubRec,
        "revolRatio": revolRatio,
        "totalAcc": totalAcc,
        "totalBal": totalBal,
        "totalRevLim": totalRevLim,
        "accOpen24": accOpen24,
        "avgBal": avgBal,
        "bcOpen": bcOpen,
        "bcRatio": bcRatio,
        "totalLim": totalLim,
        "totalRevBal": totalRevBal,
        "totalBcLim": totalBcLim,
        "totalIlLim": totalIlLim
    }


    # Get prediction from Flask endpoint
    prediction = get_prediction(data)

    # Display the result
   # st.write(f"Prediction: {prediction['prediction']}")



