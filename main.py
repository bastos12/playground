import streamlit as st

from helpers.selection import ElementChoice
from models.model import Plotting, DataEngeenering
from utils.constantes import TARGET
from utils.bdd import connection_db, get_dataset_dataframe
from utils.function import plot_metrics, encodage_target, corr_df, detect_colinearite, check_multi_col


def intro():
    st.set_page_config(
        page_title="ML Playground",
        layout="wide",
        initial_sidebar_state="auto"
    )
    conn = connection_db()
    return conn

def header(connexion):
    select_dataset = st.sidebar.selectbox('Choisir votre dataset', ['diabete_inde', 'moons', 'vin'])
    df = get_dataset_dataframe(select_dataset, connexion)
    targetor = DataEngeenering(df=df)
    return df, targetor.target_type, select_dataset

def sidebar(target_type):
    selector = ElementChoice(model_type=target_type)
    test_size = st.sidebar.number_input('Pourcentage du test set', min_value=0.09, max_value=0.51, value=0.33, step=0.01, help="Valeur en pourcentage")
    st.sidebar.write("le jeu de donnée de test sera de ", round(test_size * 100), " %")
    grid_search_auto = st.sidebar.toggle('Super recherche ?', value=False, help="La super recherche passera en revue toutes les combinaison possibles d'algorithme et d'hyperparamètre en selectionnant le meilleur combo pour vous")
    if grid_search_auto is False:
        st.sidebar.header("Choix des paramètres", divider=True)
        choice_algo = st.sidebar.selectbox('Selectionner algorithme', selector.user_interface['algorithme'])
        Xval = st.sidebar.toggle('Cross-validation ?', value=False, help='Activer pour effectuer des validations croisées, par default 5')
        params_model_dict = selector.create_input_from_model(choice_algo, target_type)
        return choice_algo, params_model_dict, Xval, test_size, grid_search_auto
    else:
        choice_algo, params_model_dict, Xval = None, None, False
        return choice_algo, params_model_dict, Xval, test_size, grid_search_auto

def analyse(data, model_name, target_type, hyperparams, select_dataset, grid_search_auto, test_size=0.35):
    # instances
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    if grid_search_auto is False:
        predictor = Plotting(
            features=X,
            target=y,
            model_name=model_name,
            target_type=target_type,
            hyperparams=hyperparams,
            select_dataset=select_dataset,
            test_size=test_size
        )
        result = predictor.prediction
    else:
        predictor = Plotting(
            features=X,
            target=y,
            model_name=model_name,
            target_type=target_type,
            hyperparams=hyperparams,
            select_dataset=select_dataset,
            GridSearch=grid_search_auto,
            test_size=test_size
        )
        result = predictor.prediction

    # construction UI/UX
    with st.expander("Dataframe"):
        st.dataframe(data)
    with st.expander("Analyse descritive"):
        st.dataframe(data.describe())
        st.write("Le jeu de donnée contient", data.isna().sum().sum(), " valeur(s) manquante(s)")
    with st.expander("Analyse des correlations"):
        col_1, col_2 = st.columns(2)
        with col_1:
            df_encoded = encodage_target(data)
            df_corred = corr_df(df_encoded)
            st.data_editor(
                df_corred,
                column_config={
                    "Correlation with target": st.column_config.ProgressColumn(
                        help="Correlation",
                        width='medium',
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                },
                hide_index=False,
            )
        with col_2:
            taux_corr = st.number_input('taux de colinearité entre deux variables', min_value=0.2, max_value=0.99, value=0.4, step=0.02, help="Taux de colinéarité accepté")
            df_colin = detect_colinearite(df_encoded, taux_corr)
            st.dataframe(df_colin)
            colin = check_multi_col(df_colin)
            if len(colin) != 0:
                st.error(f"Attention, il semble avoir des variables présentant plusieurs colinearités {colin}")
            else:
                st.error("Attention, vérifier si les paires de variables ci-dessus sont utilisées ensemble dans le model")

    if isinstance(result, dict) and len(result) == 2:
        st.caption(result['info'])
        st.caption(result['erreur'])

    else:
        # gestion des metrics differentes
        if isinstance(result, dict) and len(result) == 6:
            with st.expander("Metrics d'évaluation du modèle"):
                plot_metrics(predictor.metrics.metrics_calcul)
        else:
            pass

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

        # saving
        st.markdown('Souhaitez-vous sauvegarder votre model ?')
        save_model = st.button('Sauvegarder')
        if save_model:
            st.caption('Votre model a été sauvegardé')
            st.balloons()
    return None

def footer(connexion):

    connexion.close()
    return None


if __name__ == '__main__':
    conn = intro()
    data, target_type, select_dataset = header(connexion=conn)
    algo, hyperparams, Xval, test_size, grid_search_auto = sidebar(target_type=target_type)
    analyse(
        data=data,
        model_name=algo,
        target_type=target_type,
        hyperparams=hyperparams,
        select_dataset=select_dataset,
        grid_search_auto=grid_search_auto,
        test_size=test_size,
    )
    footer(connexion=conn)