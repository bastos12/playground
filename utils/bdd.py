import psycopg2
import pandas as pd
from decouple import config

from utils.constantes import get_data

def connection_db():
    conn = psycopg2.connect(
        host=config('HOST'),
        database=config('DATABASE'),
        user=config('USER'),
        password=config('PASSWORD')
    )
    return conn

def get_dataset_dataframe(name_dataset, db):
    if name_dataset == 'diabete_inde':
        sql_query = "SELECT * FROM diabete_inde;"
        cursor = db.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)

    elif name_dataset == 'vin':
        sql_query = "SELECT * FROM vin;"
        cursor = db.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)

    else:
        DATA_X, DATA_Y = get_data()
        df = pd.DataFrame({
            'feature1': DATA_X[:, 0],
            'feature2': DATA_X[:, 1],
            'target': DATA_Y
        })
    return df