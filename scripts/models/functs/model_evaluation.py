from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

def postResample_class(y_true, y_preds):
    """
    Given a vector with true results and the predictions of the model, 
    returns the confusion matrix, accuracy, kappa and a report(recall and recap) as a list. 
    """    
    # check the metrics with a confusion matrix
    confusion_matrix = pd.crosstab(y_true, y_preds, rownames=['Real'], colnames=['Pred'])
    print(confusion_matrix)
    print('')

    # print the accuracy
    accuracy = sum(1 for x,y in zip(y_preds, y_true) if x == y) / len(y_true)
    print("The accuracy of that model is: ", round(accuracy,4))

    # kappa 
    kappa = cohen_kappa_score(y1 = y_true, y2 = y_preds)
    print('The kappa of that model is: ', round(kappa,4))
    print('')

    # recall and recap
    report = classification_report(y_true=y_true, y_pred=y_preds) 
    print(report)
    
    results = [confusion_matrix, accuracy, kappa, report]
    return results


###############################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

def plot_errors_building(df, y_true, y_pred):
    """
    Given a dataframe, the true values and the predictions for the building, 
    return a scatter plot highlighting the errors
    """
    errors = y_true != y_pred
    data_plot = pd.DataFrame({
        'LONG': df['LONGITUDE'],
        'LAT': df['LATITUDE'],
        'err': errors
    })

    sns.scatterplot(x='LONG', y='LAT', hue='err', data=data_plot, 
                    palette=['lightgrey','red'], x_jitter=True, y_jitter=True)
    plt.title('Plotting building errors')
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.show