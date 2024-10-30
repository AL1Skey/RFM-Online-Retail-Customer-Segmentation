# import libraries
import pandas as pd
import os
import time
import sys
from datetime import datetime 

# Airflow
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup
# Joblib
import joblib
# sqlalchemy
from sqlalchemy import create_engine

#Import Pyspark and initialize spark
# from pyspark.sql import SparkSession
# from pyspark.sql.types import StringType,TimestampType 
# spark = SparkSession.builder.app('dataframe').getOrCreate()

# RClass untuk Recency

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

## FMClass untuk Frequency dan Monetary value

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1


def feedsql():
    username = os.environ['POSTGRES_USER']
    password = os.environ['POSTGRES_PASSWORD']
    database = os.environ['POSTGRES_DB']
    host = os.environ['POSTGRES_HOST']
    
    pg_url = f'postgresql+psycopg2://{username}:{password}@{host}/{database}'
    engine = create_engine(pg_url)
    conn = engine.connect()
    
    cwd = os.getcwd()
    filelist = os.listdir(os.path.join(cwd,'data'))
    file = [i for i in filelist if '.xlsx' in i][0]
    df = pd.read_excel(f'/opt/airflow/data/{file}')
    # raise Exception(df)
    df.to_sql('dirty',conn,index=False,if_exists='replace')
    
def feedcsv():
    username = os.environ['POSTGRES_USER']
    password = os.environ['POSTGRES_PASSWORD']
    database = os.environ['POSTGRES_DB']
    host = os.environ['POSTGRES_HOST']
    
    pg_url = f'postgresql+psycopg2://{username}:{password}@{host}/{database}'
    engine = create_engine(pg_url)
    conn = engine.connect()
    
    df = pd.read_sql_query('select * from dirty',conn)
    df.to_csv('/opt/airflow/data/dirty.csv',sep=',', index=False)

def preprocessing():
    username = os.environ['POSTGRES_USER']
    password = os.environ['POSTGRES_PASSWORD']
    database = os.environ['POSTGRES_DB']
    host = os.environ['POSTGRES_HOST']
    
    pg_url = f'postgresql+psycopg2://{username}:{password}@{host}/{database}'
    engine = create_engine(pg_url)
    conn = engine.connect()
    
    df = pd.read_csv('/opt/airflow/data/dirty.csv')
    
    # Missing Value Handling
    # - = SimpleImputter(0), . = SimpleImputter(0)
    df.replace('.','0',inplace=True)
    df.replace('-','0',inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Drop value less than 0
    dropped = df.select_dtypes(include='int')
    st = ''
    for i in dropped.columns:
        st+=f' {i} < 0 or'
    st = st[:-2]
    drop_index = dropped.query(st).index
    df.drop(index=drop_index, inplace=True)
    
    #Convert To Int
    df[df.columns[0]].replace(to_replace=r'[A-Za-z]',value='',regex=True,inplace=True)
    df[df.columns[0]] = df[df.columns[0]].astype(int)
    
    #Convert to proper data type
    df[df.columns[4]] = pd.to_datetime(df[df.columns[4]])
    df[df.columns[6]] = df[df.columns[6]].astype(int).astype(str)
    
    
    # Save Cleaned Data
    df.to_csv('/opt/airflow/data/clean.csv',index=False,sep=',')
    df.to_sql('clean',conn,index=False,if_exists='replace')

def modeling():
    try:
        get_model = os.listdir(os.path.join(os.getcwd(),'models'))
        name_model = [i for i in get_model if 'pkl' in i][0]
        model = joblib.load(f'/opt/airflow/models/{name_model}')
        
        df1 = pd.read_csv('/opt/airflow/data/rfm.csv')
        df1['cluster'] = model.predict(df1)
        quartiles = df1.quantile(q=[0.25,0.50,0.75])
        # mengelompokkan pelanggan ke dalam kuartil berdasarkan Recency, Frequency, dan Monetary
        rfmSeg = df1.copy()
        rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
        rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
        rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))
        
        # Menggabungkan nilai kuartil dari Recency, Frequency, dan Monetary
        rfmSeg['RFMClass'] = rfmSeg.R_Quartile.map(str) \
                                    + rfmSeg.F_Quartile.map(str) \
                                    + rfmSeg.M_Quartile.map(str)
        
        df2 = pd.read_csv('/opt/airflow/data/clean.csv')
        df = pd.merge(df1,df2,on='CustomerID')
        
        df.to_csv('/opt/airflow/data/backend.csv',index=False)
        df.to_csv('/opt/airflow/backend/data/rfm_cluster.csv',index=False)
        
        df1 = rfmSeg
        df = pd.merge(df1,df2,on='CustomerID')
        df.to_csv('/opt/airflow/data/rfm_cluster.csv',index=False)
        
    except Exception as e:
        raise Exception(e)

def creating_rfm():
    df = pd.read_csv('/opt/airflow/data/clean.csv')
    df['TotalPrice'] = df['UnitPrice']*df['Quantity']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(df.columns)
    now = datetime(2011,12,10)
    rfmTable = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (now - x.max()).days, # Recency
                                        'CustomerID': lambda x: len(x), # Frequency
                                        'TotalPrice': lambda x: x.sum()}) # Monetary Value
    rfmTable.rename(columns={'InvoiceDate': 'recency',
                            'CustomerID': 'frequency',
                            'TotalPrice': 'monetary_value'}, inplace=True)
    
    
    print(rfmTable.columns)
    rfmTable.to_csv('/opt/airflow/data/rfm.csv')       

default_args = {
    'owner': 'adam',
    'start_date': datetime(2022, 12, 24, 12, 00)
}

with DAG('data_pipeline',
         description='Data Processing',
         schedule_interval='0 0 1 * *',
         default_args=default_args,
         catchup=False) as dag:
    # Pass data to sql
    tosql = PythonOperator(
        task_id='tosql',
        python_callable=feedsql
    )
    # Pass data to csv
    tocsv = PythonOperator(
        task_id='tocsv',
        python_callable=feedcsv
    )
    # Cleaning Data
    cleaning = PythonOperator(
        task_id='cleaning',
        python_callable=preprocessing
    )
    # RFM
    rfm = PythonOperator(
        task_id='rfm',
        python_callable=creating_rfm
    )
    
    # Clustering
    cluster = PythonOperator(
        task_id='segmentation',
        python_callable=modeling
    )
    
    tosql >> tocsv >> cleaning >> rfm >> cluster
    

# with DAG('data_clustering',
#          description='Data Processing',
#          schedule_interval='30 6 * * *',
#          default_args=default_args,
#          catchup=False) as dag:
#     # Pass data to sql
#     tosql = PythonOperator(
#         task_id='tosql',
#         python_callable=feedsql
#     )
#     # Pass data to csv
#     tocsv = PythonOperator(
#         task_id='tocsv',
#         python_callable=feedcsv
#     )
#     # Cleaning Data
#     cleaning = PythonOperator(
#         task_id='cleaning',
#         python_callable=preprocessing
#     )
#     clustering = PythonOperator(
#         task_id='cleaning',
#         python_callable=preprocessing
#     )
    
#     tosql >> tocsv >> cleaning >> clustering