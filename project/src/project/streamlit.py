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
        st.write("Full response:", prediction)

        # Check if 'prediction' key exists in the response
        if 'prediction' in prediction:
            st.write(f"Prediction: {prediction['prediction']}")
        else:
            st.error("Prediction key not found in response.")
    else:
        st.error(f"Failed to get prediction. Status code: {response.status_code}")
    
    return response.json()

# Streamlit UI
st.title("Loan Prediction App")

# Creating input fields for each feature
amount = st.number_input("Loan Amount", min_value=0)
term = st.selectbox("Term", ["36 months", "60 months"])
rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, step=0.01)
payment = st.number_input("Payment", min_value=0.0)
grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
length = st.selectbox("Length of Employment", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                                               "6 years", "7 years", "8 years", "9 years", "10+ years"])
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
income = st.number_input("Annual Income", min_value=0)
verified = st.selectbox("Verification Status", ["Not Verified", "Source Verified", "Verified"])
reason = st.selectbox("Loan Reason", ["credit_card", "debt_consolidation", "other", "home_improvement", "car","major_purchase","medical","moving","vacation","house","renewable_energy"])
state = st.selectbox("State", ["CA", "PA", "NY", "TX", "FL", "OH", "WA", "VA", "MN", "NC", "RI", "LA", "WI", "NJ", "AK", "AL", "AR", "AZ", "CO", "CT", "DC", "DE", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "MA", "MD", "ME", "MI", "MO", "MS", "MT", "ND", "NE", "NH", "NM", "NV", "OK", "OR", "SC", "SD", "TN", "UT", "VT", "WV", "WY"])
debtIncRat = st.number_input("Debt-to-Income Ratio", min_value=0.0, step=0.01)
delinq2yr = st.number_input("Delinquencies in Last 2 Years", min_value=0)
inq6mth = st.number_input("Inquiries in Last 6 Months", min_value=0)
openAcc = st.number_input("Number of Open Accounts", min_value=0)
pubRec = st.number_input("Number of Public Records", min_value=0)
revolRatio = st.number_input("Revolving Line Utilization Rate", min_value=0.0, max_value=1.0, step=0.01)
totalAcc = st.number_input("Total Number of Credit Lines", min_value=0)
totalBal = st.number_input("Total Current Balance", min_value=0.0)
totalRevLim = st.number_input("Total Revolving Credit Limit", min_value=0.0)
accOpen24 = st.number_input("Accounts Opened in Last 24 Months", min_value=0)
avgBal = st.number_input("Average Account Balance", min_value=0.0)
bcOpen = st.number_input("Bankcard Accounts Open", min_value=0)
bcRatio = st.number_input("Bankcard Utilization Rate", min_value=0.0, max_value=100.0, step=0.01)
totalLim = st.number_input("Total Credit Limit", min_value=0.0)
totalRevBal = st.number_input("Total Revolving Balance", min_value=0.0)
totalBcLim = st.number_input("Total Bankcard Limit", min_value=0.0)
totalIlLim = st.number_input("Total Installment Limit", min_value=0.0)

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
        # ... (Include all other features)
    }

    # Get prediction from Flask endpoint
    prediction = get_prediction(data)

    # Display the result
    st.write(f"Prediction: {prediction['prediction']}")
