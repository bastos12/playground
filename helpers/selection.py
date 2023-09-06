"""
Classe d'aide pour la selection des modeles
"""

import streamlit as st
from utils.constantes import STRUCTURE

class ElementChoice:

    def __init__(self, model_type):
        self.structure = STRUCTURE
        self.user_interface = {
            'algorithme': self.choice_algorithme(model_type=model_type)
        }

    @staticmethod
    def _reverse_split_name_structure_algorithme(algo_str: str) -> str:
        new_str = algo_str.replace(' ', '_')
        return new_str

    def _split_name_structure_algorithme(self, model_type) -> list:
        algo = list(self.structure[model_type].keys())
        name_algo = []
        for i in algo:
            name_algo.append(i.replace('_', ' '))
        return name_algo

    def _get_list_of_param_model(self, model_name: str, target_type: str):
        modal_name = self._reverse_split_name_structure_algorithme(model_name)
        return self.structure[target_type][modal_name]['hyperparameters'].keys()

    def _get_input_params_from_model(self, model_name: str, target):
        return list(self._get_list_of_param_model(model_name=model_name, target_type=target))

    def create_input_from_model(self, model_name: str, type_target: str):
        model = self._reverse_split_name_structure_algorithme(model_name)
        params = self._get_input_params_from_model(model_name=model_name, target=type_target)
        input_values = {}

        for i in params:
            values_param = self.structure[type_target][model]['hyperparameters'][i]['values']
            selected_value = None
            if isinstance(values_param[0], str):
                selected_value = st.sidebar.selectbox(f'Paramètre {i}', values_param)
            elif isinstance(values_param[0], int):
                selected_value = st.sidebar.slider(
                    label=f'Paramètre {i}',
                    min_value=min(values_param),
                    max_value=max(values_param),
                    value=values_param[0]
                )
            input_values[i] = selected_value

        return input_values

    def choice_algorithme(self, model_type) -> list:
        return self._split_name_structure_algorithme(model_type)

