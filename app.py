import joblib as jb
import pandas as pd
import streamlit as st
import xgboost
from sklearn.preprocessing import OneHotEncoder

st.title('Intelligent Mobile Money Fraud Prediction')

model = jb.load('fraud_detection_model.pkl')

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


def process_file(file):
    if file is not None:
        df = pd.read_csv(file)
        st.subheader("Uploaded Data")
        st.write(df.head())

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder_df = encoder.fit(df[['type']])
        type_encoded = encoder_df.transform(df[['type']])
        type_encoded_df = pd.DataFrame(type_encoded, columns=encoder_df.get_feature_names_out(['type']))
        type_encoded_df.index = df.index 

        df = pd.concat([df, type_encoded_df], axis=1)
        df = df.drop(['type', 'nameOrig', 'nameDest'], axis=1)

        if 'isFraud' in df.columns:
            X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
        else:
            X = df

        prediction = model.predict(X)
        df['prediction'] = prediction

        st.subheader("Prediction Results")

        df['Result'] = df['prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')
        st.write(df[['prediction', 'Result']])

        st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")
    else:
        st.warning('Please upload a CSV file.')



process_file(uploaded_file)

