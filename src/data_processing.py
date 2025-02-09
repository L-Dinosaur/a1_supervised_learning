def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def inverse_normalize(series):
    if (series == 0).any():
        raise ValueError(f'Dividing by zero error at position: {series[series==0].index}')
    return 1 / series


def summarize_df(df):
    print('*************** Summarizing Data Frame ***************')
    for col in df.columns:
        if df[col].unique().shape[0] > 10:
            print(f'{col}: {sorted(df[col].unique().tolist()[:10])}...')
        else:
            print(f'{col}: {sorted(df[col].unique().tolist())}')
    print('*************** Summary End ***************')

