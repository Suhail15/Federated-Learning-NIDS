import flwr as fl
import numpy as np

class MomentumAggregator(fl.server.strategy.FedAvg):
    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.velocity = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)[0]

        if self.velocity is None:
            self.velocity = [np.zeros_like(w) for w in aggregated_weights]

        self.velocity = [self.momentum * v + (1 - self.momentum) * w for v, w in zip(self.velocity, aggregated_weights)]
        new_weights = [w + v for w, v in zip(aggregated_weights, self.velocity)]
        return new_weights, {}

if __name__ == "__main__":
    strategy = MomentumAggregator(momentum=0.9)
    # Simplified server start up with just the necessary parameters
    fl.server.start_server(server_address="0.0.0.0:8080", config={"num_rounds": 10}, strategy=strategy)
