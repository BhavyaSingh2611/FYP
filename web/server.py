"""
Chess ML Arena — Play against trained neural network models.

Flask server that loads PyTorch checkpoints and exposes a JSON API
for the Svelte frontend.
"""
import atexit
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.engine
import torch
from flask import Flask, jsonify, request, send_from_directory

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.agents.learning_agent import LearningAgent
from src.agents.random_agent import RandomAgent
from src.device import get_device

app = Flask(__name__, static_folder="client/dist")

RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEVICE = get_device(verbose=True)

MODEL_NAMES = [
    "convnet", "resnet", "square_transformer",
    "piece_transformer", "gcn", "gat",
]

agent_cache: dict[str, LearningAgent] = {}
games: dict[str, "GameSession"] = {}

try:
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
except Exception:
    engine = None


def cleanup_engine():
    if engine:
        engine.quit()


atexit.register(cleanup_engine)


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def scan_runs() -> list[dict]:
    results = []
    if not RUNS_DIR.exists():
        return results
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        models = [
            p.stem for p in sorted(run_dir.glob("*.pt"))
            if p.stem in MODEL_NAMES
        ]
        if models:
            results.append({"name": run_dir.name, "models": models})
    return results


def load_agent(model_name: str, run_name: str) -> LearningAgent:
    cache_key = f"{run_name}/{model_name}"
    if cache_key in agent_cache:
        return agent_cache[cache_key]

    checkpoint_path = RUNS_DIR / run_name / f"{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(str(CONFIG_PATH))
    config.model.backbone = model_name
    config.model.head = "dual"

    model = create_model(config.model)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

    agent = LearningAgent(
        model=model,
        encoder=encoder,
        device=DEVICE,
        temperature=0.0,
    )
    agent_cache[cache_key] = agent
    return agent


class GameSession:
    def __init__(self, game_id: str, agent, player_color: bool,
                 model_name: str, run_name: str):
        self.id = game_id
        self.board = chess.Board()
        self.agent = agent
        self.player_color = player_color
        self.model_name = model_name
        self.run_name = run_name
        self.move_history: list[dict] = []
        self.evals: list[dict | None] = []
        self.resigned = False
        self.evals.append(self._evaluate())

    def _evaluate(self) -> dict | None:
        if engine is None:
            return None
        infos = engine.analyse(
            self.board, chess.engine.Limit(depth=16), multipv=3,
        )
        if not infos:
            return None
        top = infos[0]
        score = top["score"].white()
        best = top.get("pv", [None])[0]
        ev = {}
        if score.is_mate():
            ev["score"] = None
            ev["mate"] = score.mate()
        else:
            ev["score"] = score.score() / 100.0
            ev["mate"] = None
        if best:
            ev["best_move"] = best.uci()
            ev["best_move_san"] = self.board.san(best)
        else:
            ev["best_move"] = None
            ev["best_move_san"] = None

        lines = []
        for info in infos:
            s = info["score"].white()
            pv = info.get("pv", [])
            line = {}
            if s.is_mate():
                line["score"] = None
                line["mate"] = s.mate()
            else:
                line["score"] = s.score() / 100.0
                line["mate"] = None
            san_moves = []
            tmp = self.board.copy()
            for m in pv[:6]:
                san_moves.append(tmp.san(m))
                tmp.push(m)
            line["moves"] = " ".join(san_moves)
            lines.append(line)
        ev["lines"] = lines
        return ev

    def to_dict(self) -> dict:
        legal_moves = []
        if not self.board.is_game_over() and not self.resigned:
            legal_moves = [m.uci() for m in self.board.legal_moves]

        last_move = None
        if self.move_history:
            uci = self.move_history[-1]["uci"]
            last_move = {"from": uci[:2], "to": uci[2:4]}

        status = "playing"
        result = None
        if self.resigned:
            status = "resigned"
            result = "0-1" if self.player_color == chess.WHITE else "1-0"
        elif self.board.is_checkmate():
            status = "checkmate"
            result = "0-1" if self.board.turn == chess.WHITE else "1-0"
        elif self.board.is_stalemate():
            status = "stalemate"
            result = "1/2-1/2"
        elif self.board.is_insufficient_material():
            status = "draw"
            result = "1/2-1/2"
        elif self.board.is_fifty_moves():
            status = "draw"
            result = "1/2-1/2"
        elif self.board.is_repetition():
            status = "draw"
            result = "1/2-1/2"

        return {
            "id": self.id,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "player_color": "white" if self.player_color == chess.WHITE else "black",
            "legal_moves": legal_moves,
            "move_history": self.move_history,
            "status": status,
            "result": result,
            "is_check": self.board.is_check(),
            "last_move": last_move,
            "model_name": self.model_name,
            "run_name": self.run_name,
            "eval": self.evals[-1] if self.evals else self._evaluate(),
        }

    def push_move(self, move: chess.Move) -> None:
        side = "white" if self.board.turn == chess.WHITE else "black"
        san = self.board.san(move)
        self.board.push(move)
        eval_result = self._evaluate()
        self.evals.append(eval_result)
        self.move_history.append({
            "uci": move.uci(), "san": san, "by": side, "eval": eval_result,
        })

    def is_player_turn(self) -> bool:
        return self.board.turn == self.player_color

    def make_ai_move(self) -> str | None:
        if self.board.is_game_over():
            return None
        move = self.agent.get_move(self.board)
        self.push_move(move)
        return move.uci()


# ---- API ----

@app.route("/api/runs")
def api_runs():
    return jsonify({"runs": scan_runs()})


@app.route("/api/game/new", methods=["POST"])
def api_new_game():
    data = request.json or {}
    model_name = data.get("model")
    run_name = data.get("run")
    player_color_str = data.get("player_color", "white")

    if not model_name or not run_name:
        return jsonify({"error": "model and run are required"}), 400

    try:
        if model_name == "random":
            agent = RandomAgent()
        else:
            agent = load_agent(model_name, run_name)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load model: {exc}"}), 500

    player_color = chess.WHITE if player_color_str == "white" else chess.BLACK
    game_id = str(uuid.uuid4())
    session = GameSession(game_id, agent, player_color, model_name, run_name)
    games[game_id] = session

    if not session.is_player_turn():
        session.make_ai_move()

    return jsonify(session.to_dict())


@app.route("/api/game/<game_id>")
def api_game_state(game_id):
    session = games.get(game_id)
    if not session:
        return jsonify({"error": "Game not found"}), 404
    return jsonify(session.to_dict())


@app.route("/api/game/<game_id>/move", methods=["POST"])
def api_make_move(game_id):
    session = games.get(game_id)
    if not session:
        return jsonify({"error": "Game not found"}), 404
    if session.board.is_game_over() or session.resigned:
        return jsonify({"error": "Game is over"}), 400
    if not session.is_player_turn():
        return jsonify({"error": "Not your turn"}), 400

    data = request.json or {}
    move_uci = data.get("move")
    if not move_uci:
        return jsonify({"error": "move is required"}), 400

    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return jsonify({"error": "Invalid move format"}), 400

    if move not in session.board.legal_moves:
        return jsonify({"error": "Illegal move"}), 400

    session.push_move(move)

    if not session.board.is_game_over():
        session.make_ai_move()

    return jsonify(session.to_dict())


@app.route("/api/game/<game_id>/eval")
def api_eval(game_id):
    session = games.get(game_id)
    if not session:
        return jsonify({"error": "Game not found"}), 404
    ev = session._evaluate()
    return jsonify({"eval": ev})


@app.route("/api/game/<game_id>/resign", methods=["POST"])
def api_resign(game_id):
    session = games.get(game_id)
    if not session:
        return jsonify({"error": "Game not found"}), 404
    session.resigned = True
    return jsonify(session.to_dict())


# ---- Static serving ----

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    print(f"\nAvailable runs: {scan_runs()}")
    print(f"Device: {DEVICE}")
    print(f"\nChess ML Arena → http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
