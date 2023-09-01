"""
Constantes utilisées dans le playground
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import streamlit as st

STRUCTURE = {
    'classification': {
        'Regression_Logistique': {
            'model': LogisticRegression(),
            'hyperparameters': {
                'penalty': {
                    'type': str,
                    'values': ['l1', 'l2', 'elasticnet', 'None'],
                },
                'C': {
                    'type': int,
                    'values': [1, 5, 10],
                },
                'solver': {
                    'type': str,
                    'values': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                }
            }
        },
        'Arbre_Decision': {
            'model': DecisionTreeClassifier(),
            'hyperparameters': {}
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {}
        },
        'KNeighbours_Classifier': {
            'model': KNeighborsClassifier(),
            'hyperparameters': {}
        },
        'SVC': {
            'model': SVC(),
            'hyperparameters': {}
        },
    },
    'regression': {
        'Regression_Lineaire': {
            'model': LinearRegression(),
            'hyperparameters': {}
        },
        'Regression_Ridge': {
            'model': Ridge(),
            'hyperparameters': {
                'alpha': {
                    'type': int,
                    'values': [0, 1, 100],
                },
            }
        },
    }
}

TARGET = 'target'


@st.cache_data(show_spinner=True)
def get_data():
    x, y = make_moons(n_samples=1000)
    return x, y
