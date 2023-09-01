import streamlit as st

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