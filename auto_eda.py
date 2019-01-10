import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataframe_readcsv(csvFileName):
    '''
    Helper function that gives a dataframe of a given csv filename

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    df = pd.read_csv(csvFileName)
    return df

def dataframe_shape(df):
    '''
    Helper function that gives a shape of a given dataframe

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    print('\n')
    print('='*80)

    print('Shape:')
    print(df.shape)

def dataframe_head(df):
    '''
    Helper function that gives a datatypes of a given dataframe

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    print('\n')
    print('='*80)

    print('Head:')
    print(df.head())

def dataframe_tail(df):
    '''
    Helper function that gives a tail of a given dataframe

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    print('\n')
    print('='*80)

    print('Tail:')
    print(df.tail())

def dataframe_dtypes(df):
    '''
    Helper function that gives a head of a given dataframe

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    print('\n')
    print('='*80)

    print('Data Types:')
    print(df.dtypes)

def dataframe_info(df):
    '''
    Helper function that gives a info of a given dataframe

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    print('\n')
    print('='*80)

    print('Info:')
    print(df.info())

def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed
    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    plt.show()

def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot|
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)
    plt.show()

def init_numeric_types(df):
    '''
    Helper function that gives a quick summary of quantattive data

    Arguments
    =========
    dataframe: pandas dataframe

    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    print('\n')
    print('Numeric Data Types:')
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for o in numeric_columns:
        try:
            print('\n')
            print(f'{o}')
            quantitative_summarized(df, y=o)
        except:
            continue

def init_object_types(df):
    '''
    Helper function that gives a quick summary of a given column of categorical data

    Arguments
    =========
    dataframe: pandas dataframe

    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    print('\n')
    print('Object Data Types:')
    object_columns = df.select_dtypes(include=[np.object]).columns.tolist()
    for o in object_columns:
        print('\n')
        print(f'{o}')
        categorical_summarized(df, x=o)

def init_dataframe(csvFileName, shape=False, head=False, tail=False, dtypes=False, info=False, summaryObject=False, summaryNumeric=False):
    '''
    Helper function that gives a a high level summary of a given dataframe

    Arguments
    =========
    dataframe: pandas dataframe
    '''
    df = pd.read_csv(csvFileName)

    if shape == True:
        dataframe_shape(df)

    if head == True:
        dataframe_head(df)

    if tail == True:
        dataframe_tail(df)

    if dtypes == True:
        dataframe_dtypes(df)

    if info == True:
        dataframe_info(df)

    if summaryObject == True:
        init_object_types(df)

    if summaryNumeric == True:
        init_numeric_types(df)
