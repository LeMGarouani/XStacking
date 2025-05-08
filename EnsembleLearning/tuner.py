from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.svm import SVC,SVR
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

def TuneClassifier(classifier, X, Y):
    # Choisir le modèle et l'espace de recherche des hyperparamètres en fonction du classificateur
    if classifier == "svc":
        model = SVC()
        param_distributions = {
            'C': uniform(0.1, 10),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': randint(2, 5),
            'gamma': ['scale', 'auto'],
            'coef0': uniform(0, 1)
        }

    elif classifier == "xgb":
        model = xgb.XGBClassifier(use_label_encoder=False, objective='multi:softmax', num_class=len(np.unique(Y)))
        param_distributions = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 0.3),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(1, 3)
        }
    else:
        raise ValueError("meta-Classifier not supported. Choose 'svc' or 'xgb'.")

    # Configurer RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # Entraîner le modèle avec la recherche aléatoire
    random_search.fit(X, Y)

    # Afficher les meilleurs hyperparamètres
    #print(f"Best parameters for {classifier}: ", random_search.best_params_)

    # Retourner le modèle avec les meilleurs hyperparamètres
    return random_search.best_estimator_





def TuneRegressor(regressor, X, Y):
    # Choisir le modèle et l'espace de recherche des hyperparamètres en fonction du classificateur
    if regressor == "svr":
        model = SVR()
        param_distributions = {
            'kernel': ['linear','poly'],
            'C': [0.1, 1, 10, 100]
            #'gamma': ['scale', 'auto', 0.1, 1, 10],
            #'degree': [2, 3, 4],
            #'coef0': [0.01, 0.1, 0.5],
        }
        param_dist = {
            'C': uniform(loc=0.1, scale=9.9),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'epsilon': uniform(loc=0.01, scale=0.09)
        }

    elif regressor == "xgbr":
        model = xgb.XGBRegressor(eval_metric=mean_squared_error)
        param_distributions = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 0.3),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(1, 3)
        }
    else:
        raise ValueError("meta-Classifier not supported. Choose 'svc' or 'xgb'.")

    # Configurer RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=3,
        scoring='neg_mean_squared_error',
        cv=5,

        verbose=3

    )#random_state=42,

    print("start research...")
    # Entraîner le modèle avec la recherche aléatoire
    random_search.fit(X, Y)

    # Afficher les meilleurs hyperparamètres
    #print(f"Best parameters for {classifier}: ", random_search.best_params_)
    print("research done...")
    # Retourner le modèle avec les meilleurs hyperparamètres
    return random_search.best_estimator_