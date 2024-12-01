import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# Удаляет ненужные столбцы
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

# Подготоваливает числовые столбцы
class ProcessNumericColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            median = X[col].median()
            X[col] = X[col].fillna(median)
        return X

# Подготоваливает категориальные столбцы
class ProcessCategoricalColumns(BaseEstimator, TransformerMixin):
    mappings = {
        'indrel_1mes': {
            '1.0': '1',
            '2.0': '2',
            '3.0': '3',
            '4.0': '4'
        }
    }
    
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].replace([pd.NA, None], 'unknown').astype(str)
            X[col] = X[col].str.strip()
            X[col] = X[col].replace(['NA', '', 'nan'], 'unknown')
            if col in self.mappings:
                X[col] = X[col].map(self.mappings[col])
        return X

# Подготоваливает столбцы с датами
class ProcessDateColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
        return X        

# Подготавливает столбцы с булевыми значениями
class ProcessBooleanColumns(BaseEstimator, TransformerMixin):
    mappings = {
        'indresi': {'S': True, 'N': False},
        'indext': {'S': True, 'N': False},
        'indfall': {'S': True, 'N': False},
        'sexo': {'H': True, 'V': False}
    }

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in self.mappings:
                X[col] = X[col].map(self.mappings[col])
            X[col] = X[col].fillna(False)
            X[col] = X[col].astype(int)
        return X

# Удаляет выбросы
class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns, threshold=1.5):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        bounds = {
            col: (X[col].quantile(0.25) - self.threshold * (X[col].quantile(0.75) - X[col].quantile(0.25)),
                  X[col].quantile(0.75) + self.threshold * (X[col].quantile(0.75) - X[col].quantile(0.25)))
            for col in self.columns
        }
        for col, (lower, upper) in bounds.items():
            X = X[(X[col] >= lower) & (X[col] <= upper)]
        return X

# Преобразовывает категориальные столбцы
class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col].astype(str))
        return X
