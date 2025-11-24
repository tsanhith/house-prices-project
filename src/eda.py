import pandas as pd

def load_data(path='data/train.csv'):
    df = pd.read_csv(path)
    return df

def quick_summary(df):
    print('shape:', df.shape)
    print('\\nmissing per column:')
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    print('\\nnumeric describe:')
    print(df.describe().T)

if __name__ == '__main__':
    df = load_data()
    quick_summary(df)
