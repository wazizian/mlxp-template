import mlxp
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_id', type=int, help='Log ID')
    args = parser.parse_args()
    log_id = args.log_id

    parent_log_dir = './logs/'
    reader = mlxp.Reader(parent_log_dir)
    query = f"info.logger.log_id == {log_id}"
    results = reader.filter(query)
    for key in reader.searchable.index:
        try:
            val = results[0][key]
        except KeyError:
            continue
        print(f"{key}: {val}")

if __name__ == '__main__':
    main()
