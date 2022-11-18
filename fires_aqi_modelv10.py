import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance

def partialDependencePlots(gbc, X):
    explainer = shap.TreeExplainer(gbc)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)
    shap.dependence_plot('Length', shap_values, X, interaction_index=None)
    shap.dependence_plot('temp', shap_values, X, interaction_index=None)
    shap.dependence_plot('precip', shap_values, X, interaction_index=None)
    shap.dependence_plot('dratio', shap_values, X, interaction_index=None)
    shap.dependence_plot('available_green', shap_values, X, interaction_index=None)
    shap.dependence_plot('income', shap_values, X, interaction_index=None)
    shap.dependence_plot('AcresBurned', shap_values, X, interaction_index=None)
    shap.dependence_plot('nearby_maxdensity', shap_values, X, interaction_index=None)

    return

def confusionMatirx(gbc, X_test, y_test):
    y_pred = gbc.predict(X_test)
    confMatrix = confusion_matrix(y_test,y_pred)
    print("\nConfusion Matrix")
    print("%d |  %d " % (confMatrix[1][1], confMatrix[1][0]))
    print("--------")
    print("0%d | %d" % (confMatrix[0][1], confMatrix[0][0]))

    return

def plot_fi(fi, X):
    importances = fi['importances_mean']
    imp_errors = fi['importances_std']
    scale = np.max(importances)
    y_pos = np.arange(X.columns.size)
    sortme = np.argsort(importances)
    plt.figure()
    plt.barh(y_pos, importances[sortme]/scale, alpha=0.3, xerr = imp_errors[sortme]/scale)
    plt.yticks(y_pos, X.columns[sortme])
    plt.xlabel('Feature importance')
    plt.show()

    return

def featureImportance(gbc, X, y, use):
    print("\nFeature Importance in Gradient Boosting Classifier Model")
    print("-----------------------------------------------------------------")
    r = permutation_importance(gbc, X, y, n_repeats=30, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{use[i]:<20}\t"
                    f"{r.importances_mean[i]:.3f}"
                    f" +/- {r.importances_std[i]:.3f}")
    plot_fi(r, X)

    return

def displayAUC(y_train, y_test, X_train, X_test, gbc):
    print("\nArea Under the Curve for Gradient Boosting Classifier Model")
    print("---------------------------------------------------------------------")
    print("Training Data: " + str(roc_auc_score(y_train, gbc.predict_proba(X_train)[:,1])))
    print("Test Data: " + str(roc_auc_score(y_test, gbc.predict_proba(X_test)[:,1])))

def applyGBC(df3, use):
    X = df3[use]
    y = df3['dangerous']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, stratify = y, train_size = 0.6)
    gbc = GradientBoostingClassifier(n_estimators=20, max_depth=5, learning_rate=0.1,
                                    min_samples_leaf=5, random_state=1).fit(X_train, y_train)
    print("\nAccuracy of Gradient Boosting Classifier Model")
    print("-----------------------------------------------------")
    print("Training Data: " + str(gbc.score(X_train, y_train)))
    print("Test Data: " + str(gbc.score(X_test, y_test)))

    displayAUC(y_train, y_test, X_train, X_test, gbc)
    featureImportance(gbc, X, y, use)
    confusionMatirx(gbc, X_test, y_test)

    partialDependencePlots(gbc, X)

    return

def getFeatures(df):
    cols = ['AcresBurned','Length','available_green',
            'nearby_maxdensity','dratio','income', 'temp', 'precip','dangerous']
    df2 = df[~(df['Name'] == 'Camp Fire')][cols]
    use = ['AcresBurned','Length','available_green',
            'nearby_maxdensity','dratio','income', 'temp', 'precip']
    df3 = df2[cols].replace([np.inf, -np.inf], np.nan)
    df3 = df3.fillna(0)

    return df3, use

def main():
    df = pd.read_csv('fires_aqi_more6.csv')
    print("Size: " + str(df.shape))
    print("Columns: " + str(df.columns.values))
    df3, use = getFeatures(df)
    applyGBC(df3, use)

main()
