from time import sleep

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers.selection import ElementChoice
from helpers.metrics import MetricsScorer
import altair as alt
from sklearn.model_selection import GridSearchCV
from stqdm import stqdm

class DataEngeenering:

    def __init__(self, df):
        self.df = df
        self.target_type = self._get_target_type()

    def _get_target_type(self):
        if isinstance(self.df['target'][0], str):
            target_type = 'classification'
        elif isinstance(self.df['target'][0], np.int64):
            target_type = 'regression'
        else:
            target_type = 'classification'
        return target_type

class DataSplitter:
    def __init__(self, features, target, test_size=0.35):
        self.X = features
        self.target = target
        self.test_size = test_size
        self.random_state = 42

    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.target,
            test_size=self.test_size
        )
        return X_train, X_test, y_train, y_test


class DataPredict(DataSplitter, ElementChoice):

    def __init__(self, features, target, target_type, model_name, hyperparams, grid_search_auto=False, test_size=0.35):
        DataSplitter.__init__(
            self,
            features=features,
            target=target,
            test_size=test_size,
        )
        ElementChoice.__init__(
            self,
            model_type=target_type
        )
        self.hyperparams = hyperparams
        if grid_search_auto is False:
            self.model = self._get_model_instance(
                model_name=model_name,
                target_type=target_type
            )
        else:
            self.model = self._get_models(
                target_type=target_type,
            )

    def _get_model_instance(self, model_name, target_type):
        instance_name = self._reverse_split_name_structure_algorithme(model_name)
        model = self.structure[target_type][instance_name]['model']
        return model.set_params(**self.hyperparams)

    def _get_models(self, target_type):
        models = []
        model = None
        param_dict = {}
        for i in self.structure[target_type]:
            for info in self.structure[target_type][i]:
                if info == 'model':
                    model = self.structure[target_type][i][info]
                if info == 'hyperparameters':
                    parameters = {}
                    for params in self.structure[target_type][i][info]:
                        parameters[params] = self.structure[target_type][i][info][params]['values']
                    param_dict = parameters
                models.append({'model': model, 'params': param_dict})
        return models

    def _fitting(self):
        X_train, X_test, y_train, y_test = self.split()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return {
            'split': (X_train, X_test, y_train, y_test),
            'fitted_model': self.model.fit(X_train, y_train)
        }

    def predict(self):
        try:
            obj = self._fitting()
            y_pred_train = obj['fitted_model'].predict(obj['split'][0])
            y_pred_test = obj['fitted_model'].predict(obj['split'][1])
            return dict(
                X_train=obj['split'][0],
                X_test=obj['split'][1],
                y_pred_train=y_pred_train,
                y_pred_test=y_pred_test,
                y_train=obj['split'][2],
                y_test=obj['split'][3]
            )
        except ValueError as e:
            return {
                'info': "Les combinaisons de paramètres ne sont pas corrects",
                'erreur': f"Détails: {e}"
            }

    def grid_search_prediction(self, X, y, cv=5):
        result = []
        try:
            with st.spinner('Recherche des meilleurs combinaisons algorithme/hyperparamètres'):
                for _mod in self.model:
                    if _mod['model'] and _mod['params']:
                        try:
                            clf = GridSearchCV(_mod['model'], _mod['params'], cv=cv)
                            clf.fit(X, y)
                            result.append({
                                'model_name': _mod['model'],
                                'best_score': clf.best_score_,
                                'best_params': clf.best_params_
                            })
                        except:
                            pass
            return result

        except:
            return {
                'info': "Erreur lors de la recherche des paramètres et de l'algorihme",
                'erreur': f"Détails: a venir"
            }


class Plotting(DataPredict):

    def __init__(
        self,
        features,
        target,
        model_name,
        target_type,
        hyperparams,
        select_dataset,
        GridSearch=False,
        test_size=0.35
    ):
        super().__init__(
            features=features,
            target=target,
            target_type=target_type,
            model_name=model_name,
            hyperparams=hyperparams,
            grid_search_auto=GridSearch,
            test_size=test_size
        )

        self.GridSearch = GridSearch
        if GridSearch:
            self.prediction = self.grid_search_prediction(
                X=features,
                y=target
            )
        else:
            self.prediction = self.predict()

        self.graph_general = True if select_dataset != 'moons' else False
        self.metrics = MetricsScorer(
            target_type=target_type,
            dict_value=self.prediction
        )

    def plotting_values_training(self):
        if self.graph_general:
            df = pd.DataFrame({
                'X': self.prediction['X_train'][:, 0],
                'Classe': self.prediction['y_train']
            })
            df_1 = pd.DataFrame({
                'X': self.prediction['X_train'][:, 0],
                'Classe': self.prediction['y_pred_train']
            })
        else:
            df = pd.DataFrame({
                'X': self.prediction['X_train'][:, 0],
                'Y': self.prediction['X_train'][:, 1],
                'Classe': self.prediction['y_train']
            })
            df_1 = pd.DataFrame({
                'X': self.prediction['X_train'][:, 0],
                'Y': self.prediction['X_train'][:, 1],
                'Classe': self.prediction['y_pred_train']
            })
        return df, df_1

    def plotting_values_test(self):
        if self.graph_general:
            df = pd.DataFrame({
                'X': self.prediction['X_test'][:, 0],
                'Classe': self.prediction['y_test']
            })
            df_1 = pd.DataFrame({
                'X': self.prediction['X_test'][:, 0],
                'Classe': self.prediction['y_pred_test']
            })
        else:
            df = pd.DataFrame({
                'X': self.prediction['X_test'][:, 0],
                'Y': self.prediction['X_test'][:, 1],
                'Classe': self.prediction['y_test']
            })
            df_1 = pd.DataFrame({
                'X': self.prediction['X_test'][:, 0],
                'Y': self.prediction['X_test'][:, 1],
                'Classe': self.prediction['y_pred_test']
            })
        return df, df_1

    def _get_palette(self, df):
        custom_palette = alt.Color(
            df.columns[-1],
            scale=alt.Scale(range=['orangered', 'lightgreen'])
        )
        return custom_palette

    def create_chart_scatter(self, df):
        df[df.columns[-1]] = df[df.columns[-1]].astype(str)
        palette = self._get_palette(df)
        c = alt.Chart(df).mark_circle().encode(
            x=df.columns[0],
            y=df.columns[1],
            color=palette,
            tooltip=list(df.columns)
        )
        st.altair_chart(c, use_container_width=True)
