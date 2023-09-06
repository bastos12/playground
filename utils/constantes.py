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

PARAM_HELPER = {
    'interpretation': {
        'r2_score': 'intervalle evaluation du score r2',
        'recall': 'intervalle evaluation du score recall',
        'f1-score': 'intervalle evaluation du score f1',
        'accuracy': 'intervalle evaluation du score acc',
        'mse': 'intervalle evaluation du score mse',
    },
    'explication': {
        'r2_score': 'Ceci est une aide pour interpreter le r2_score',
        'recall': 'Ceci est une aide pour interpreter le recall',
        'f1-score': 'Ceci est une aide pour interpreter le f1-score',
        'accuracy': 'Ceci est une aide pour interpreter le accuracy',
        'mse': 'Ceci est une aide pour interpreter le mse',
    }
}

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
            'hyperparameters': {
                'n_estimators': {
                    'type': int,
                    'values': [10, 100, 500, 1000]
                },
                'criterion': {
                    'type': str,
                    'values': ["gini", "entropy", "log_loss"],
                },
                'max_depth': {
                    'type': str,
                    'values': [1, 5, 10, 50],
                },
                'class_weight': {
                    'type': str,
                    'values': ['balanced', 'balanced_subsample']
                }
            }
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
