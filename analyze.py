import matplotlib.pyplot as plt
import mlxp
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
import cvxpy as cp
from sklearn.base import BaseEstimator, RegressorMixin

class LinearUpperBound(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        n, d = X.shape
        
        # Define cvxpy variables
        w = cp.Variable(d)  # Coefficients
        if self.fit_intercept:
            b = cp.Variable()   # Intercept
        slack = cp.Variable(n, nonneg=True)  # Slack variables
        
        # Objective: Minimize L1 error
        objective = cp.Minimize(cp.sum(slack))
        
        # Constraints: Ensure y <= Xw + b + slack
        if self.fit_intercept:
            constraints = [X @ w + b >= y, slack >= X @ w + b - y]
        else:
            constraints = [X @ w >= y, slack >= X @ w - y]
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CVXOPT)
        
        # Store the results
        self.coef_ = w.value
        if self.fit_intercept:
            self.intercept_ = b.value
        
        return self

    def predict(self, X):
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_

def moving_average(x, w):
    res = np.cumsum(x, dtype=float)
    res[w:] = res[w:] - res[:-w]
    return res[w - 1 :] / w

def fill_missing(mlxp_df, keys):
    for datadict in mlxp_df:
        for key in keys:
            if key not in datadict:
                datadict.update({key: ""})

def analyze_MLP():
    parent_log_dir = "./logs/"
    reader = mlxp.Reader(parent_log_dir)

    query = "(info.status == 'COMPLETE' | info.status == 'RUNNING') & config.model._target_ == 'tasks.simple_classification.models.build_MLP'"

    results = reader.filter(query)
    keys = results.keys()
    fill_missing(results, keys)
    print(keys)
    print(f"Found {len(results)} results with ids {results[:]['info.logger.log_id']}")

    data_keys = [key for key in results.keys() if key.startswith("config.dataset")]
    model_keys = [key for key in results.keys() if key.startswith("config.model")]
    optim_keys = [key for key in results.keys() if key.startswith("config.optimizer")]
    avg_keys = [key for key in results.keys() if key.startswith("config.averager")]
    loss_keys = [key for key in results.keys() if key.startswith("config.loss")]
    train_keys = [key for key in results.keys() if key.startswith("config.train")]

    grouped_results = results.groupby(data_keys + optim_keys + loss_keys)
    val_loss_key = "train_metrics.val/loss"
    window = 10
    for i, (group_key, df) in enumerate(grouped_results.items()):
        df = df.toPandas(lazy=False)
        df = df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
        print(f"Group {i}:")
        print(f"Found {len(df)} runs with keys {df[:]['info.logger.log_id']}")
        print(f"Config for group {i}:")
        for k, v in zip(data_keys + optim_keys + loss_keys, group_key):
            print(f"  {k}: {v}")

        grouped_df = df.groupby("config.model.hidden_sizes")

        plt.figure()
        plt.title(f"Group {i}")
        for optim_key, sub_df in grouped_df:
            if len(sub_df) > 1:
                print(f"Multiple runs found for {group_key} and {optim_key}")
            val_loss = sub_df.iloc[0].loc[val_loss_key]
            smoothed_val_loss = moving_average(val_loss, window)
            t = np.arange(len(smoothed_val_loss)) + window // 2
            plt.plot(t, smoothed_val_loss, label=optim_key)
            plt.plot(val_loss, alpha=0.2, color=plt.gca().lines[-1].get_color())
            avg_key = tuple([results[0][k] for k in avg_keys])
        plt.legend()
        plt.show()

def compute_MLP_hitting_time():
    parent_log_dir = "./logs/"
    reader = mlxp.Reader(parent_log_dir)

    query = "(info.status == 'COMPLETE' | info.status == 'RUNNING') & config.model._target_ == 'tasks.simple_classification.models.build_MLP'"

    results = reader.filter(query, result_format="pandas")
    results = results.fillna("").applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    keys = results.keys()
    print(keys)
    print(f"Found {len(results)} results with ids {results[:]['info.logger.log_id']}")

    data_keys = [key for key in results.keys() if key.startswith("config.dataset")]
    model_keys = [key for key in results.keys() if key.startswith("config.model")]
    optim_keys = [key for key in results.keys() if key.startswith("config.optimizer")]
    loss_keys = [key for key in results.keys() if key.startswith("config.loss")]
    train_keys = [key for key in results.keys() if key.startswith("config.train")]

    grouped_results = results.groupby(data_keys + model_keys + loss_keys)
    val_acc_key = "train_metrics.val/acc"
    window = 1
    plt.figure()
    abs_thresholds = [0., 0.01, 0.05, 0.1]

    for i, (group_key, df) in enumerate(grouped_results):
        print(f"Group {i}:")
        print(f"Found {len(df)} runs with keys {df[:]['info.logger.log_id']}")
        print(f"Config for group {i}:")
        for k, v in zip(data_keys + model_keys + train_keys + loss_keys, group_key):
            print(f"  {k}: {v}")
        
        hidden_sizes = df.iloc[0].loc["config.model.hidden_sizes"]
        if hidden_sizes[0] < 8:
            continue

        learning_rates = [[] for _ in abs_thresholds]
        hitting_times = [[] for _ in abs_thresholds]
        grouped_df = df.groupby("config.optimizer.learning_rate")

        for optim_key, sub_df in grouped_df:
            if len(sub_df) > 1:
                print(f"Multiple runs found for {group_key} and {optim_key}")
            val_acc = sub_df.iloc[0].loc[val_acc_key]
            smoothed_val_acc = moving_average(val_acc, window)
            max_smoothed_val_acc = np.max(smoothed_val_acc)
            print("Max accuracy: ", max_smoothed_val_acc)
            for i, acc_abs_thresh in enumerate(abs_thresholds):
                converged = np.argwhere(smoothed_val_acc >= 1.0 - acc_abs_thresh)
                print("Threshold: ", 1.0 - acc_abs_thresh)
                if len(converged) > 0:
                    print("Converged at: ", converged[0])
                    hitting_time = converged[0]
                    learning_rate = sub_df.iloc[0].loc["config.optimizer.learning_rate"]
                    print("Learning rate: ", learning_rate)
                    learning_rates[i].append(learning_rate)
                    hitting_times[i].append(hitting_time)

        for i, acc_abs_thresh in enumerate(abs_thresholds):
            inv_lr = 1 / np.array(learning_rates[i])
            lr = np.array(learning_rates[i])
            htimes = np.array(hitting_times[i]).flatten()
            
            estimator = LinearRegression(fit_intercept=True)
            # estimator.fit(np.stack((lr, inv_lr), axis=1), np.log(hitting_times[i]))
            sample_weight = inv_lr ** 3
            sample_weight = sample_weight / np.sum(sample_weight)
            estimator.fit(inv_lr.reshape(-1, 1), np.log(htimes), sample_weight=sample_weight)

            inv_lrs = np.linspace(min(inv_lr), max(inv_lr), 100)
            lrs = 1 / inv_lrs
            # predicted = estimator.predict(np.stack((lrs, inv_lrs), axis=1))
            predicted = estimator.predict(inv_lrs.reshape(-1, 1))
            scatter_label = f"Accuracy threshold = {(1 - acc_abs_thresh) * 100}%"
            plot_label = f"Linear fit for accuracy threshold = {(1 - acc_abs_thresh) * 100}%"
            sc = plt.scatter(inv_lr, np.log(htimes), alpha=1, s=25, marker="x", label=scatter_label)
            color = sc.get_facecolors()[0]
            plt.plot(inv_lrs, predicted, alpha=0.2, label=plot_label, color=color)
    plt.legend()
    plt.xlabel("1 / Learning rate")
    plt.ylabel("Log hitting time to accuracy threshold")
    plt.title("Hitting time to accuracy threshold for MLP as a function of learning rate")
    plt.show()

if __name__ == "__main__":
    # analyze_MLP()
    compute_MLP_hitting_time()
