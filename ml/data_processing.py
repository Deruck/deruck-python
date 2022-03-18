import numpy as np

def data_split(X, y, test_size=0.3):
    """split data with test size
    
    
    Args:
        X: m x n
        y: 1 x m
    
    Returns:
        (X_train, X_test, y_train, y_test)
        
    """
    n = X.shape[0]
    train_indeces = np.random.choice(n, size=round(n * (1 - test_size)), replace=False)
    test_indeces = np.delete(np.arange(n), train_indeces)
    return X[train_indeces], X[test_indeces], y[train_indeces], y[test_indeces]