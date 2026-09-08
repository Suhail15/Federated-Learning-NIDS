"""Run one local Flower client using the capstone DNN."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/df_aug_shuff_c1.csv")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import flwr as fl
    import tensorflow as tf
    from data import load_partition
    from model import build_dnn_model

    tf.keras.utils.set_random_seed(args.seed)
    x_train, y_train, x_val, y_val = load_partition(
        args.data, args.client_id, args.num_clients, args.seed
    )
    model = build_dnn_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class DNNClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=int(config.get("epochs", 10)),
                      batch_size=int(config.get("batch_size", 128)), verbose=2)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
            return float(loss), len(x_val), {"accuracy": float(accuracy)}

    fl.client.start_client(server_address=args.server, client=DNNClient().to_client())


if __name__ == "__main__":
    main()
