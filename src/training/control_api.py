"""
Live training control API.

Lightweight Flask server that runs in a background daemon thread during
training, exposing endpoints to inspect status and modify hyperparameters
on-the-fly.
"""

import logging
import threading

from flask import Flask, jsonify, request

from .trainer import Trainer

LOGGER = logging.getLogger(__name__)


class TrainingController:
    """Wraps a :class:`Trainer` reference and provides a Flask app for live control."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.lock = threading.Lock()
        self.app = self._create_app()

    # ------------------------------------------------------------------
    # Flask app factory
    # ------------------------------------------------------------------

    def _create_app(self) -> Flask:
        app = Flask(__name__)

        @app.after_request
        def cors(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return response

        @app.route("/api/status")
        def status():
            cfg = self.trainer.training_cfg
            return jsonify(
                {
                    "epoch": self.trainer.epoch,
                    "global_step": self.trainer.global_step,
                    "best_loss": self.trainer.best_loss,
                    "training_running": not self.trainer._should_stop,
                    "hyperparams": {
                        "lr": cfg.learning_rate,
                        "batch_size": cfg.batch_size,
                        "policy_loss_weight": cfg.policy_loss_weight,
                        "value_loss_weight": cfg.value_loss_weight,
                        "grad_clip_max_norm": cfg.grad_clip_max_norm,
                    },
                }
            )

        @app.route("/api/stats")
        def stats():
            return jsonify(
                {
                    "epoch": self.trainer.epoch,
                    "global_step": self.trainer.global_step,
                    "best_loss": self.trainer.best_loss,
                    "history": self.trainer.stats_history,
                }
            )

        @app.route("/api/hyperparams", methods=["POST"])
        def update_hyperparams():
            data = request.json or {}
            if not data:
                return jsonify({"error": "empty request body"}), 400

            allowed = {
                "learning_rate",
                "policy_loss_weight",
                "value_loss_weight",
                "grad_clip_max_norm",
                "gradient_accumulation_steps",
            }
            unknown = set(data.keys()) - allowed
            if unknown:
                return jsonify({"error": f"unknown keys: {sorted(unknown)}"}), 400

            with self.lock:
                cfg = self.trainer.training_cfg

                if "learning_rate" in data:
                    lr = float(data["learning_rate"])
                    cfg.learning_rate = lr
                    for pg in self.trainer.optimizer.param_groups:
                        pg["lr"] = lr
                    LOGGER.info(f"Updated learning_rate → {lr}")

                if "policy_loss_weight" in data:
                    w = float(data["policy_loss_weight"])
                    cfg.policy_loss_weight = w
                    if hasattr(self.trainer.loss_fn, "policy_weight"):
                        self.trainer.loss_fn.policy_weight = w  # type: ignore
                    LOGGER.info(f"Updated policy_loss_weight → {w}")

                if "value_loss_weight" in data:
                    w = float(data["value_loss_weight"])
                    cfg.value_loss_weight = w
                    if hasattr(self.trainer.loss_fn, "value_weight"):
                        self.trainer.loss_fn.value_weight = w  # type: ignore
                    LOGGER.info(f"Updated value_loss_weight → {w}")

                if "grad_clip_max_norm" in data:
                    v = float(data["grad_clip_max_norm"])
                    cfg.grad_clip_max_norm = v
                    LOGGER.info(f"Updated grad_clip_max_norm → {v}")

                if "gradient_accumulation_steps" in data:
                    v = int(data["gradient_accumulation_steps"])
                    cfg.gradient_accumulation_steps = v
                    LOGGER.info(f"Updated gradient_accumulation_steps → {v}")

            return jsonify({"status": "ok", "updated": list(data.keys())})

        @app.route("/api/stop", methods=["POST"])
        def stop():
            self.trainer._should_stop = True
            LOGGER.info("Graceful stop requested via control API")
            return jsonify({"status": "stopping"})

        return app


def start_control_server(trainer: Trainer, port: int = 5050) -> threading.Thread:
    """Start the control API in a background daemon thread.

    Returns the thread so the caller can inspect it if needed.
    """
    controller = TrainingController(trainer)

    def _run():
        LOGGER.info(f"Training control API listening on http://0.0.0.0:{port}")
        controller.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, name="training-control-api", daemon=True)
    thread.start()
    return thread
