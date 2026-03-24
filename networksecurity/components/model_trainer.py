import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,GradientBoostingClassifier
import mlflow



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config: ModelTrainerConfig = model_trainer_config
            self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        except Exception as e:  
            raise NetworkSecurityException(e,sys)

    def track_mlflow(self,best_model,classification_metric):
            with mlflow.start_run():
                f1_score = classification_metric["f1_score"]

                precision_score = classification_metric["precision_score"]
                recall_score = classification_metric["recall_score"]

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)

                mlflow.sklearn.log_model(best_model, "model")

    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            logging.info("Training the model")
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            }
            model_report:dict = evaluate_models(X_train = x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            #Track The ML Flow



            y_train_pred = best_model.predict(x_train)

            classification_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            self.track_mlflow(best_model=best_model,classification_metric=classification_metric)


            y_test_pred = best_model.predict(x_test)
            test_classification_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)
            self.track_mlflow(best_model=best_model,classification_metric=test_classification_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            model = NetworkModel(preprocessor_object=preprocessor,model_object = best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=model)

            save_object("final_models/best_model.pkl",best_model)



            ## Model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        train_metric_artifact=classification_metric,
                                                        test_metric_artifact=test_classification_metric,
                                                        model_name=best_model_name)
            logging.info(f"Model training completed. Best model: {best_model_name} with train metric: {classification_metric} and test metric: {test_classification_metric}")   
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")
            logging.info("Loading transformed training and testing data")
            X_train = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            X_test = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)


            x_train,y_train = X_train[:,:-1],X_train[:,-1]
            x_test,y_test = X_test[:,:-1],X_test[:,-1]
            

            model = self.train_model(x_train,y_train,x_test,y_test)
            return model

        except Exception as e:
            raise NetworkSecurityException(e,sys)