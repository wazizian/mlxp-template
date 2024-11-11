import mlxp
import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w):
    res = np.cumsum(x, dtype=float)
    res[w:] = res[w:] - res[:-w]
    return res[w - 1:] / w

def fill_missing(mlxp_df, keys):
    for datadict in mlxp_df:
        for key in keys:
            if key not in datadict:
                datadict.update({key: ""})

def analyze():
    parent_log_dir = './logs/'
    reader = mlxp.Reader(parent_log_dir)

    query = "info.status == 'COMPLETE'"
    results = reader.filter(query)
    keys = results.keys()
    fill_missing(results, keys)
    print(keys)

    print(f"Found {len(results)} results with ids {results[:]['info.logger.log_id']}")
    data_keys = [key for key in results.keys() if key.startswith('config.dataset')]
    model_keys = [key for key in results.keys() if key.startswith('config.model')]
    optim_keys = [key for key in results.keys() if key.startswith('config.optimizer')]
    avg_keys = [key for key in results.keys() if key.startswith('config.averager')]
    loss_keys = [key for key in results.keys() if key.startswith('config.loss')]
    train_keys = [key for key in results.keys() if key.startswith('config.train')]
    grouped_results = results.groupby(data_keys + model_keys + loss_keys)
    val_loss_key = 'train_metrics.val/loss'
    avg_val_loss_key = 'train_metrics.avg_val/loss'
    window = 200
    for i, (group_key, df) in enumerate(grouped_results.items()):
        print(f"Group {i}:")
        print(f"Found {len(df)} runs with keys {df[:]['info.logger.log_id']}")
        print(f"Config for group {i}:")
        for k, v in zip(data_keys + model_keys + loss_keys, group_key):
            print(f"  {k}: {v}")

        grouped_df = df.groupby(optim_keys)

        plt.figure()
        plt.title(f"Group {i}")
        for optim_key, sub_df in grouped_df.items():
            if len(sub_df) > 1:
                print(f"Multiple runs found for {group_key} and {optim_key}")
            val_loss = sub_df[0][val_loss_key]
            smoothed_val_loss = moving_average(val_loss, window)
            t = np.arange(len(smoothed_val_loss)) + window // 2
            plt.plot(t, smoothed_val_loss, label=optim_key)
            plt.plot(val_loss, alpha=0.2, color=plt.gca().lines[-1].get_color())
            avg_key = tuple([results[0][k] for k in avg_keys])
            avg_val_loss = sub_df[0][avg_val_loss_key]
            smoothed_avg_val_loss = moving_average(avg_val_loss, window)
            t = np.arange(len(smoothed_val_loss)) + window // 2
            plt.plot(t, smoothed_avg_val_loss, label=str(optim_key) + " " + str(avg_key))
            plt.plot(avg_val_loss, alpha=0.2, color=plt.gca().lines[-1].get_color())
        plt.legend()
        plt.show()

if __name__ == '__main__':
    analyze()


