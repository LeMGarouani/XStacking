import copy
from sklearn.metrics import accuracy_score,mean_squared_error
import shap
import numpy as np
from tuner import TuneClassifier, TuneRegressor
class StackingClassifier:
    def __init__(self, base_learners, meta_learner):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.fitted_base_learners = []

    def fit(self, X, y,ponderation, mode='normal'):
        self.mode = mode
        self.ponderation=ponderation
        meta_features = []  # to store the normal mode meta-features
        meta_features_xai = []  # to store the xai mode meat-features
        self.fitted_base_learners = []  # to store the trained models
        self.fitted_base_explainers = []  # to store the explainers of each trained base-learner
        self.accuracies=[]
        self.weights=[]
        # Train base learners and generate cross-validated predictions to serve as meta-features
        for base_learner in self.base_learners:
            fitted_learner = base_learner.fit(X, y)
            cloned_learner = copy.deepcopy(fitted_learner)
            self.fitted_base_learners.append(cloned_learner)
            predictions = fitted_learner.predict(X)
            print(base_learner, ":", accuracy_score(y, predictions))
            self.accuracies.append(accuracy_score(y, predictions))
            if self.mode == 'normal':
                meta_features.append(predictions)

            if self.mode == 'xai':
                if X.shape[0]>1000:
                    print("in...")
                    X_summary = shap.kmeans(X, 50).data
                    explainer = shap.KernelExplainer(base_learner.predict_proba, X_summary)
                    #or shap_values = explainer.shap_values( shap.sample(X_test, 100) )
                else:
                    explainer = shap.KernelExplainer(base_learner.predict_proba, X)
                shap_values = explainer.shap_values(X)
                shap_values_pred_class = np.array(
                    [shap_values[i][:, class_idx] for i, class_idx in enumerate(predictions)])

                meta_features_xai.append(shap_values_pred_class)
                cloned_explainer = copy.deepcopy(explainer)
                self.fitted_base_explainers.append(cloned_explainer)
        #Calculate weights
        total_accuracy = sum(self.accuracies)
        self.weights = [acc / total_accuracy for acc in self.accuracies]
        print("weights: ",self.weights)
        # Stack meta-features horizontally
        if self.mode == 'normal':
            meta_features = np.array(meta_features).T
        if self.mode == 'xai':
            if ponderation==True:
                print('in ponderation...',len(meta_features_xai),len(meta_features_xai[0]))
                #print(meta_features_xai[0])
                for i in range(len(meta_features_xai)):
                    meta_features_xai[i]=meta_features_xai[i]*self.weights[i]
            #print(len(meta_features_xai[0]))
            #print(meta_features_xai[0])
            meta_features_xai = np.hstack(meta_features_xai)

        # Train the meta-learner on the meta-features
        if self.mode == 'normal':
            self.meta_learner = TuneClassifier(self.meta_learner, meta_features, y)
            # self.meta_learner.fit(meta_features, y)
        if self.mode == 'xai':
            # self.meta_learner.fit(meta_features_xai, y)
            self.meta_learner = TuneClassifier(self.meta_learner, meta_features_xai, y)

    def predict(self, X):
        # Generate meta-features for new data
        if self.mode == 'normal':
            meta_features = [learner.predict(X) for learner in self.fitted_base_learners]
            meta_features = np.array(meta_features).T
            return self.meta_learner.predict(meta_features)

        if self.mode == 'xai':
            meta_features_xai_ = []
            w=0
            for base_learner, base_explainer in zip(self.fitted_base_learners, self.fitted_base_explainers):
                prediction = base_learner.predict(X)
                shap_values = base_explainer.shap_values(X)
                shap_values_pred_class = np.array(
                    [shap_values[i][:, class_idx] for i, class_idx in enumerate(prediction)])
                if self.ponderation == True:
                    meta_features_xai_.append(shap_values_pred_class*self.weights[w])
                    w+=1
                else:
                    print('no ponteration!')
                    meta_features_xai_.append(shap_values_pred_class)

            meta_features_xai = np.hstack(meta_features_xai_)
            return self.meta_learner.predict(meta_features_xai)






class StackingRegressor:
    def __init__(self, base_learners, meta_learner):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.fitted_base_learners = []

    def fit(self, X, y,ponderation, mode='normal'):
        self.mode = mode
        self.ponderation=ponderation
        meta_features = []  # to store the normal mode meta-features
        meta_features_xai = []  # to store the xai mode meat-features
        self.fitted_base_learners = []  # to store the trained models
        self.fitted_base_explainers = []  # to store the explainers of each trained base-learner
        self.accuracies=[]
        self.weights=[]
        # Train base learners and generate cross-validated predictions to serve as meta-features
        for base_learner in self.base_learners:
            fitted_learner = base_learner.fit(X, y)
            cloned_learner = copy.deepcopy(fitted_learner)
            self.fitted_base_learners.append(cloned_learner)
            predictions = fitted_learner.predict(X)
            print(base_learner, ":", mean_squared_error(y, predictions))
            self.accuracies.append(mean_squared_error(y, predictions))
            if self.mode == 'normal':
                meta_features.append(predictions)

            if self.mode == 'xai':
                explainer = shap.Explainer(base_learner.predict, X)
                try:
                    shap_values = explainer.shap_values(X)
                except AttributeError:
                    print('in finally...')
                    shap_values = explainer(X)
                    shap_values = shap_values.values

                meta_features_xai.append(shap_values)
                cloned_explainer = copy.deepcopy(explainer)
                self.fitted_base_explainers.append(cloned_explainer)
        #Calculate weights
        total_accuracy = sum(self.accuracies)
        self.weights = [acc / total_accuracy for acc in self.accuracies]
        print("weights: ",self.weights)
        # Stack meta-features horizontally
        if self.mode == 'normal':
            meta_features = np.array(meta_features).T
        if self.mode == 'xai':
            if ponderation==True:
                print('in ponderation...',len(meta_features_xai),len(meta_features_xai[0]))
                #print(meta_features_xai[0])
                for i in range(len(meta_features_xai)):
                    meta_features_xai[i]=meta_features_xai[i]*self.weights[i]
            #print(len(meta_features_xai[0]))
            #print(meta_features_xai[0])
            meta_features_xai = np.hstack(meta_features_xai)

        # Train the meta-learner on the meta-features
        if self.mode == 'normal':
            self.meta_learner = TuneRegressor(self.meta_learner, meta_features, y)
            print("tuning done ;)")
            # self.meta_learner.fit(meta_features, y)
        if self.mode == 'xai':
            # self.meta_learner.fit(meta_features_xai, y)
            self.meta_learner = TuneRegressor(self.meta_learner, meta_features_xai, y)

    def predict(self, X):
        # Generate meta-features for new data
        if self.mode == 'normal':
            meta_features = [learner.predict(X) for learner in self.fitted_base_learners]
            meta_features = np.array(meta_features).T
            return self.meta_learner.predict(meta_features)

        if self.mode == 'xai':
            meta_features_xai_ = []
            w=0
            for base_learner, base_explainer in zip(self.fitted_base_learners, self.fitted_base_explainers):
                prediction = base_learner.predict(X)
                try:
                    shap_values = base_explainer.shap_values(X)
                except AttributeError:
                    print("in except of predict...")
                    shap_values = base_explainer(X)
                    shap_values = shap_values.values
                if self.ponderation == True:
                    meta_features_xai_.append(shap_values*self.weights[w])
                    w+=1
                else:
                    print('no ponteration!')
                    meta_features_xai_.append(shap_values)

            meta_features_xai = np.hstack(meta_features_xai_)
            return self.meta_learner.predict(meta_features_xai)