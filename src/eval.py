# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

# function to score dataset from Kaggle
def score_dataset(X, y, model):
    # calculate Root Mean Squared Log Error
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# learning curve for neural network
def plot_nn_history(history):
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot();
    print(f"Minimum validation loss: {(history_df['val_loss'].min())**0.5}; Epoch = {history_df['val_loss'].idxmin()}")
    print(f"Minimum training loss: {(history_df['loss'].min())**0.5}; Epoch = {history_df['loss'].idxmin()}")
          
    return history_df
      
# function to plot learning curve
def plot_learning_curve(estimator, X, y, cv, model_name):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv, scoring='neg_mean_squared_log_error')
    # convert to RMSLE
    train_scores = - train_scores
    test_scores = - test_scores
    
    # create figure
    _, axes = plt.subplots(1, figsize=(10, 6))

    axes.set_title(f"Learning Curve - {model_name}")
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Mean Square Logarithmic Error")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    
    return plt
      
#for model in models[:2]:
#    plot_learning_curve(model['model'], X_train_full_transformed, y_train_full, 5, model['model_name'])
    
#plt.show()
