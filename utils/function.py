import streamlit as st
from sklearn import preprocessing
import pandas as pd

from utils.constantes import PARAM_HELPER

def plot_metrics(values_dict: dict):
    clef = list(values_dict.keys())
    values = list(values_dict.values())
    col1, col2 = st.columns(2)
    with col1:
        for i in range(len(clef)):
            st.markdown(clef[i], help=PARAM_HELPER['explication'][clef[i]])
    with col2:
        for i in range(len(values)):
            st.markdown(values[i], help=PARAM_HELPER['interpretation'][clef[i]])

def encodage_target(df):
    data = df.copy()
    le = preprocessing.LabelEncoder()
    data['target'] = le.fit_transform(data['target'])
    return data

def corr_df(df):
    test = df.corrwith(df['target'])
    new = pd.DataFrame(test)
    new = new.rename(columns={0: 'Correlation with target'})
    new['Correlation with target'] = abs(new['Correlation with target'])
    return new

def detect_colinearite(df, value_detect=0.4):
    feature_1 = []
    feature_2 = []
    value = []
    for i in df.columns[:-1]:
        corr = df.corrwith(df[i])
        for idx, val in enumerate(abs(corr)):
            if val >= value_detect and i != corr.index[idx] and corr.index[idx] != 'target':
                feature_1.append((i))
                feature_2.append(corr.index[idx])
                value.append(round(val, 3))
    df_colin = pd.DataFrame({
        'features_1': feature_1,
        'features_2': feature_2,
        'value': value
    })
    df_colin = df_colin.drop_duplicates(subset='value')
    return df_colin

def check_multi_col(df):
    col_1 = df['features_1'].value_counts()
    col_2 = df['features_2'].value_counts()
    danger = []
    for idx, i in enumerate(col_1):
        if i > 1:
            danger.append(col_1.index[idx])
    for idx, i in enumerate(col_2):
        if i > 1:
            danger.append(col_2.index[idx])
    return list(set(danger))
