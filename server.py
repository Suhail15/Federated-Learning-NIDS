"""Flower server with FedAvg and the original experimental weight-momentum rule."""
import argparse
import flwr as fl
import numpy as np


def weighted_accuracy(metrics):
    total = sum(count for count, _ in metrics)
    return {"accuracy": sum(count * m["accuracy"] for count, m in metrics) / total} if total else {}


class FedMomentum(fl.server.strategy.FedAvg):
    """Preserves the capstone rule; this is not standard FedAvgM.

    v_t = momentum * v_(t-1) + (1 - momentum) * averaged_weights
    returned_weights = averaged_weights + v_t
    """
    def __init__(self, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.velocity = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, metrics
        weights = fl.common.parameters_to_ndarrays(parameters)
        if self.velocity is None:
            self.velocity = [np.zeros_like(w) for w in weights]
        self.velocity = [self.momentum * v + (1 - self.momentum) * w
                         for v, w in zip(self.velocity, weights)]
        updated = [w + v for w, v in zip(weights, self.velocity)]
        return fl.common.ndarrays_to_parameters(updated), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--strategy", choices=["capstone-momentum", "fedavg"], default="capstone-momentum")
    parser.add_argument("--address", default="127.0.0.1:8080")
    args = parser.parse_args()
    if args.num_clients < 2 or min(args.rounds, args.epochs, args.batch_size) < 1:
        parser.error("Use at least two clients and positive rounds, epochs, and batch size.")
    strategy_class = FedMomentum if args.strategy == "capstone-momentum" else fl.server.strategy.FedAvg
    strategy = strategy_class(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=args.num_clients, min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        on_fit_config_fn=lambda _: {"epochs": args.epochs, "batch_size": args.batch_size},
    )
    fl.server.start_server(server_address=args.address,
                         config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy)


if __name__ == "__main__":
    main()
