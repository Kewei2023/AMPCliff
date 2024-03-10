from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class ModelRegressor:
    def __init__(self, random_state=None):
        """
        init
        :param random_state: random_state
        :param regressor: 'SVM', 'RF' or 'XGBoost'
        """
        self.random_state = random_state
        self.models = self._regressor_models()
    
    def _regressor_models(self):
        """
        select regressor
        """
        models = {
            'SVM':SVR(kernel='rbf', C=100, gamma=0.1),
            'LR': LinearRegression(),
            'L1': Lasso(alpha=0.1),
            'L2': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'RF':RandomForestRegressor(random_state=self.random_state),
            'GB':GradientBoostingRegressor(random_state=self.random_state),
            'XGBoost':XGBRegressor(max_depth=7, n_estimators=200, learning_rate=0.1, use_label_encoder=False,
                             objective="reg:squarederror",random_state=self.random_state),
            'GP':GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=10)
        }
        return models
    
    def train(self, X_train, y_train):
        """
        train model
        :param X_train
        :param y_train
        """
        for model in self.models:
          self.models[model].fit(X_train, y_train)
          print(f"Training complete using {model}")
    
    def predict(self, X_test):
        """
        predict
        :param X_test
        :return: result
        """
        result = {}
        
        for model in self.models:
          result[f"{model}-pred"] = self.models[model].predict(X_test)
        return result
