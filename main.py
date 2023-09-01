import streamlit as st

from helpers.selection import ElementChoice
from models.model import Plotting, DataEngeenering
from utils.constantes import TARGET
from utils.bdd import connection_db, get_dataset_dataframe

def intro():
    st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )
    conn = connection_db()
    return conn

def header(connexion):
    select_dataset = st.sidebar.selectbox('Choisir votre dataset', ['moons', 'diabete_inde', 'vin'])
    df = get_dataset_dataframe(select_dataset, connexion)
    targetor = DataEngeenering(df=df)
    return df, targetor.target_type, select_dataset

def sidebar(target_type):
    selector = ElementChoice(model_type=target_type)
    choice_algo = st.sidebar.selectbox('Selectionner algorithme', selector.user_interface['algorithme'])
    params_model_dict = selector.create_input_from_model(choice_algo, target_type)
    return choice_algo, params_model_dict

def analyse(data, model_name, target_type, hyperparams, select_dataset, test_size=0.35):
    st.dataframe(data)
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    predictor = Plotting(
        features=X,
        target=y,
        model_name=model_name,
        target_type=target_type,
        hyperparams=hyperparams,
        test_size=test_size
    )
    result = predictor.prediction
    if isinstance(result, dict) and len(result) == 2:
        st.caption(result['info'])
        st.caption(result['erreur'])
    else:
        with st.expander("Graphiques évaluations des metrics"):
            df, df1 = predictor.plotting_values_training()
            df_test, df1_test = predictor.plotting_values_test()
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Train fold: valeurs réelles")
                predictor.create_chart_scatter(df)
            with col2:
                st.caption("Train fold: valeurs prédites")
                predictor.create_chart_scatter(df1)
            col3, col4 = st.columns(2)
            with col3:
                st.caption("Test fold: valeurs réelles")
                predictor.create_chart_scatter(df_test)
            with col4:
                st.caption("Test fold: valeurs prédites")
                predictor.create_chart_scatter(df1_test)
    return None

def footer(connexion):
    connexion.close()
    return None


if __name__ == '__main__':
    conn = intro()
    data, target_type, select_dataset = header(connexion=conn)
    algo, hyperparams = sidebar(target_type=target_type)
    analyse(
        data=data,
        model_name=algo,
        target_type=target_type,
        hyperparams=hyperparams,
        select_dataset=select_dataset
    )
    footer(connexion=conn)