from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.01
    epochs: int = 2000
    m: float = 0.0
    b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n = len(x)
        if n == 0:
            raise ValueError("Cannot fit on empty data.")

        for _ in range(self.epochs):
            y_hat = self.predict(x)
            error = y_hat - y

            dm = (2 / n) * np.sum(error * x)
            db = (2 / n) * np.sum(error)

            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

    def predict(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return self.m * x_arr + self.b

    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict(x)
        return float(np.mean((y_hat - y) ** 2))

    def to_dict(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "m": self.m,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "LinearRegressionGD":
        model = cls(
            learning_rate=float(payload.get("learning_rate", 0.01)),
            epochs=int(payload.get("epochs", 2000)),
        )
        model.m = float(payload["m"])
        model.b = float(payload["b"])
        return model
