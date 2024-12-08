import matplotlib.pyplot as plt
import mlxp
import numpy as np
from sklearn.linear_model import LinearRegression


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
    val_loss_key = "train_metrics.val/loss"
    window = 1
    abs_thresh = 0.02 # 0.001 absolute threshold
    plt.figure()

    for i, (group_key, df) in enumerate(grouped_results):
        print(f"Group {i}:")
        print(f"Found {len(df)} runs with keys {df[:]['info.logger.log_id']}")
        print(f"Config for group {i}:")
        for k, v in zip(data_keys + model_keys + train_keys + loss_keys, group_key):
            print(f"  {k}: {v}")
        
        hidden_sizes = df.iloc[0].loc["config.model.hidden_sizes"]
        if hidden_sizes[0] < 8:
            continue

        learning_rates = []
        hitting_times = []
        grouped_df = df.groupby("config.optimizer.learning_rate")

        for optim_key, sub_df in grouped_df:
            if len(sub_df) > 1:
                print(f"Multiple runs found for {group_key} and {optim_key}")
            val_loss = sub_df.iloc[0].loc[val_loss_key]
            smoothed_val_loss = moving_average(val_loss, window)
            min_smoothed_val_loss = np.min(smoothed_val_loss)
            converged = np.argwhere(smoothed_val_loss < abs_thresh)
            print("Min loss: ", min_smoothed_val_loss)
            print("Threshold: ", abs_thresh)
            if len(converged) > 0:
                print("Converged at: ", converged[0])
                hitting_time = converged[0]
                learning_rate = sub_df.iloc[0].loc["config.optimizer.learning_rate"]
                print("Converged at: ", converged[0])
                print("Learning rate: ", learning_rate)
                learning_rates.append(learning_rate)
                hitting_times.append(hitting_time)

        inv_lr = 1 / np.array(learning_rates)
        lr = np.array(learning_rates) * 0
        
        estimator = LinearRegression()
        estimator.fit(np.stack((lr, inv_lr), axis=1), np.log(hitting_times))

        inv_lrs = np.linspace(min(inv_lr), max(inv_lr), 100)
        lrs = 1 / inv_lrs
        predicted = estimator.predict(np.stack((lrs, inv_lrs), axis=1))
        plt.plot(inv_lrs, predicted, label=hidden_sizes, alpha=0.2)
        color = plt.gca().lines[-1].get_color()
        plt.scatter(inv_lr, np.log(hitting_times), color=color, alpha=1, s=5, marker="x")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compute_MLP_hitting_time()
