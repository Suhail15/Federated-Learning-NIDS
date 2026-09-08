import flwr as fl
import logging
import numpy as np
import sys

class FedMomentum(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit=0.1, min_fit_clients=2, min_available_clients=2, momentum=0.9):
        super().__init__(fraction_fit=fraction_fit, min_fit_clients=min_fit_clients, min_available_clients=min_available_clients)
        self.momentum = momentum
        self.velocity = None  
    def aggregate_fit(self, rnd, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_params is not None:
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_params)

            if self.velocity is None:
                self.velocity = [np.zeros_like(w) for w in aggregated_weights]

            self.velocity = [self.momentum * v + (1 - self.momentum) * w for v, w in zip(self.velocity, aggregated_weights)]
            
            aggregated_weights = [w + v for w, v in zip(aggregated_weights, self.velocity)]

            new_params = fl.common.ndarrays_to_parameters(aggregated_weights)

            if aggregated_metrics:
                logging.info(f"Aggregated metrics: {aggregated_metrics}")

            return new_params, aggregated_metrics 

def start_server():
    strategy = FedMomentum(momentum=0.9)
    logging.basicConfig(level=logging.DEBUG) 

    logging.info("Starting Flower server with federated averaging strategy with momentum")

    server_config = fl.server.ServerConfig(num_rounds=5)

    fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=strategy
    )

    logging.info("Flower server has stopped")

if __name__ == "__main__":
    start_server()
