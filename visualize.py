import pandas as pd

if __name__ == '__main__':
    progress = pd.read_csv('./models/test/progress.csv', delimiter=';')
    progress[['training loss', 'loss']].plot()
