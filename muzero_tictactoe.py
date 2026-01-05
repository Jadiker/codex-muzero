from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KnownBounds:
    minimum: float
    maximum: float


class MinMaxStats:
    def __init__(self, known_bounds: Optional[KnownBounds]):
        if known_bounds is None:
            self.minimum = float("inf")
            self.maximum = float("-inf")
        else:
            self.minimum = known_bounds.minimum
            self.maximum = known_bounds.maximum

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class MuZeroConfig:
    action_space_size: int = 9
    max_moves: int = 9
    discount: float = 1.0
    dirichlet_alpha: float = 0.3
    num_simulations: int = 50
    root_exploration_fraction: float = 0.25
    pb_c_base: int = 19652
    pb_c_init: float = 1.25
    num_unroll_steps: int = 5
    td_steps: int = 5
    batch_size: int = 64
    training_steps: int = 200
    training_steps_per_game: int = 10
    min_replay_size: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    action_embed_dim: int = 16
    seed: int = 0
    known_bounds: KnownBounds = field(default_factory=lambda: KnownBounds(-1.0, 1.0))

    def visit_softmax_temperature(self, num_moves: int) -> float:
        if num_moves < 2:
            return 1.0
        if num_moves < 6:
            return 0.5
        return 0.1


WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class TicTacToe:
    def __init__(self) -> None:
        self.board: List[int] = [0] * 9
        self.to_play: int = 1

    def clone(self) -> TicTacToe:
        copied = TicTacToe()
        copied.board = self.board.copy()
        copied.to_play = self.to_play
        return copied

    def legal_actions(self) -> List[int]:
        return [idx for idx, value in enumerate(self.board) if value == 0]

    def check_winner(self) -> int:
        for a, b, c in WINNING_LINES:
            total = self.board[a] + self.board[b] + self.board[c]
            if total == 3:
                return 1
            if total == -3:
                return -1
        return 0

    def is_terminal(self) -> bool:
        return self.check_winner() != 0 or all(v != 0 for v in self.board)

    def outcome_value(self, player: int) -> float:
        winner = self.check_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == player else -1.0

    def apply(self, action: int) -> float:
        if self.board[action] != 0:
            raise ValueError(f"Illegal action {action}")
        self.board[action] = self.to_play
        reward = 1.0 if self.check_winner() == self.to_play else 0.0
        self.to_play *= -1
        return reward

    def make_observation(self) -> torch.Tensor:
        board_tensor = torch.tensor(self.board, dtype=torch.int8).view(3, 3)
        current = (board_tensor == self.to_play).float()
        opponent = (board_tensor == -self.to_play).float()
        return torch.stack([current, opponent], dim=0)

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for i in range(0, 9, 3):
            rows.append(" ".join(symbols[v] for v in self.board[i : i + 3]))
        return "\n".join(rows)


@dataclass
class NetworkOutput:
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor


class RepresentationNet(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim: int, action_space_size: int, action_embed_dim: int) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(action_space_size, action_embed_dim)
        self.fc1 = nn.Linear(hidden_dim + action_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_emb = self.action_embed(action)
        x = torch.cat([hidden_state, action_emb], dim=-1)
        x = F.relu(self.fc1(x))
        next_state = F.relu(self.fc2(x))
        reward = torch.tanh(self.reward_head(next_state)).squeeze(-1)
        return next_state, reward


class PredictionNet(nn.Module):
    def __init__(self, hidden_dim: int, action_space_size: int) -> None:
        super().__init__()
        self.policy_head = nn.Linear(hidden_dim, action_space_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(hidden_state)
        value = torch.tanh(self.value_head(hidden_state)).squeeze(-1)
        return policy_logits, value


class MuZeroNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.representation = RepresentationNet(config.hidden_dim)
        self.dynamics = DynamicsNet(config.hidden_dim, config.action_space_size, config.action_embed_dim)
        self.prediction = PredictionNet(config.hidden_dim, config.action_space_size)

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        reward = torch.zeros_like(value)
        return NetworkOutput(value=value, reward=reward, policy_logits=policy_logits, hidden_state=hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        next_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_state)
        return NetworkOutput(value=value, reward=reward, policy_logits=policy_logits, hidden_state=next_state)


class Node:
    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.to_play = 1
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}
        self.hidden_state: Optional[torch.Tensor] = None
        self.reward = 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        child_value = child.value()
        if child.to_play != parent.to_play:
            child_value = -child_value
        value_score = child.reward + config.discount * min_max_stats.normalize(child_value)
    else:
        value_score = 0.0
    return prior_score + value_score


def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats) -> tuple[int, Node]:
    best_action = -1
    best_score = float("-inf")
    best_child: Optional[Node] = None
    for action, child in node.children.items():
        score = ucb_score(config, node, child, min_max_stats)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    if best_child is None:
        raise RuntimeError("No children to select from.")
    return best_action, best_child


def expand_node(
    node: Node,
    to_play: int,
    legal_actions: Sequence[int],
    network_output: NetworkOutput,
) -> None:
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = float(network_output.reward.squeeze(0).item())
    if not legal_actions:
        return
    policy_logits = network_output.policy_logits.squeeze(0)
    logits = policy_logits[list(legal_actions)]
    policy = torch.softmax(logits, dim=0).tolist()
    for action, prob in zip(legal_actions, policy):
        node.children[action] = Node(prob)


def backpropagate(
    search_path: List[Node],
    value: float,
    discount: float,
    min_max_stats: MinMaxStats,
) -> None:
    for idx in range(len(search_path) - 1, -1, -1):
        node = search_path[idx]
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        if idx > 0:
            parent = search_path[idx - 1]
            same_player = node.to_play == parent.to_play
            signed_value = value if same_player else -value
            value = node.reward + discount * signed_value


def add_exploration_noise(config: MuZeroConfig, node: Node) -> None:
    actions = list(node.children.keys())
    if not actions:
        return
    noise = torch.distributions.Dirichlet(
        torch.full((len(actions),), config.dirichlet_alpha)
    ).sample()
    frac = config.root_exploration_fraction
    for action, n in zip(actions, noise.tolist()):
        node.children[action].prior = node.children[action].prior * (1 - frac) + n * frac


def run_mcts(config: MuZeroConfig, root: Node, env: TicTacToe, network: MuZeroNetwork, device: torch.device) -> None:
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        sim_env = env.clone()
        last_action: Optional[int] = None

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            sim_env.apply(action)
            last_action = action
            search_path.append(node)
            if sim_env.is_terminal():
                break

        if sim_env.is_terminal():
            backpropagate(search_path, 0.0, config.discount, min_max_stats)
            continue

        if last_action is None:
            raise RuntimeError("Root should be expanded before running MCTS.")

        parent = search_path[-2]
        if parent.hidden_state is None:
            raise RuntimeError("Parent hidden state missing during MCTS expansion.")

        with torch.no_grad():
            action_tensor = torch.tensor([last_action], device=device, dtype=torch.long)
            network_output = network.recurrent_inference(parent.hidden_state, action_tensor)

        expand_node(node, sim_env.to_play, sim_env.legal_actions(), network_output)
        value = float(network_output.value.squeeze(0).item())
        backpropagate(search_path, value, config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, root: Node) -> int:
    temperature = config.visit_softmax_temperature(num_moves)
    actions = list(root.children.keys())
    visit_counts = [root.children[a].visit_count for a in actions]
    if not actions:
        raise RuntimeError("No actions available for selection.")
    if temperature <= 1e-6:
        return actions[visit_counts.index(max(visit_counts))]
    scaled = [count ** (1.0 / temperature) for count in visit_counts]
    total = sum(scaled)
    probs = [v / total for v in scaled]
    return random.choices(actions, weights=probs, k=1)[0]


@dataclass
class Target:
    value: float
    reward: float
    policy: List[float]


@dataclass
class BatchItem:
    observation: torch.Tensor
    actions: List[int]
    targets: List[Target]


@dataclass
class GameHistory:
    observations: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    child_visits: List[List[float]] = field(default_factory=list)
    root_values: List[float] = field(default_factory=list)
    to_play: List[int] = field(default_factory=list)

    def store_search_statistics(self, root: Node, action_space_size: int) -> None:
        sum_visits = sum(child.visit_count for child in root.children.values())
        if sum_visits == 0:
            self.child_visits.append([0.0 for _ in range(action_space_size)])
        else:
            visits = [
                (root.children[a].visit_count / sum_visits) if a in root.children else 0.0
                for a in range(action_space_size)
            ]
            self.child_visits.append(visits)
        self.root_values.append(root.value())

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, discount: float) -> List[Target]:
        targets: List[Target] = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            if current_index < len(self.root_values):
                bootstrap_index = current_index + td_steps
                value = 0.0
                for i in range(td_steps):
                    reward_index = current_index + i
                    if reward_index < len(self.rewards):
                        reward = self.rewards[reward_index]
                        if self.to_play[reward_index] != self.to_play[current_index]:
                            reward = -reward
                        value += reward * (discount**i)
                if bootstrap_index < len(self.root_values):
                    bootstrap_value = self.root_values[bootstrap_index]
                    if self.to_play[bootstrap_index] != self.to_play[current_index]:
                        bootstrap_value = -bootstrap_value
                    value += bootstrap_value * (discount**td_steps)
                last_reward = self.rewards[current_index - 1] if current_index > 0 else 0.0
                targets.append(
                    Target(value=value, reward=last_reward, policy=self.child_visits[current_index])
                )
            else:
                last_reward = self.rewards[-1] if self.rewards else 0.0
                targets.append(Target(value=0.0, reward=last_reward, policy=[]))
        return targets


class ReplayBuffer:
    def __init__(self, config: MuZeroConfig) -> None:
        self.window_size = 1000
        self.batch_size = config.batch_size
        self.buffer: List[GameHistory] = []
        self.config = config

    def save_game(self, game: GameHistory) -> None:
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self) -> List[BatchItem]:
        batch: List[BatchItem] = []
        for _ in range(self.batch_size):
            game = random.choice(self.buffer)
            index = random.randrange(len(game.observations))
            observation = game.observations[index]
            actions = game.actions[index : index + self.config.num_unroll_steps]
            if len(actions) < self.config.num_unroll_steps:
                actions = actions + [0] * (self.config.num_unroll_steps - len(actions))
            targets = game.make_target(index, self.config.num_unroll_steps, self.config.td_steps, self.config.discount)
            batch.append(BatchItem(observation=observation, actions=actions, targets=targets))
        return batch

    def __len__(self) -> int:
        return len(self.buffer)


def play_game(config: MuZeroConfig, network: MuZeroNetwork, device: torch.device) -> GameHistory:
    env = TicTacToe()
    history = GameHistory()
    num_moves = 0

    while not env.is_terminal() and num_moves < config.max_moves:
        observation = env.make_observation()
        history.observations.append(observation)
        history.to_play.append(env.to_play)

        root = Node(0.0)
        with torch.no_grad():
            network_output = network.initial_inference(observation.unsqueeze(0).to(device))
        expand_node(root, env.to_play, env.legal_actions(), network_output)
        add_exploration_noise(config, root)
        run_mcts(config, root, env, network, device)

        history.store_search_statistics(root, config.action_space_size)
        action = select_action(config, num_moves, root)
        reward = env.apply(action)

        history.actions.append(action)
        history.rewards.append(reward)
        num_moves += 1

    return history


def loss_for_step(
    output: NetworkOutput,
    target: Target,
    device: torch.device,
) -> torch.Tensor:
    value_target = torch.tensor(target.value, device=device)
    reward_target = torch.tensor(target.reward, device=device)
    value_loss = F.mse_loss(output.value.squeeze(0), value_target)
    reward_loss = F.mse_loss(output.reward.squeeze(0), reward_target)
    if target.policy:
        target_policy = torch.tensor(target.policy, device=device)
        log_probs = F.log_softmax(output.policy_logits.squeeze(0), dim=0)
        policy_loss = -(target_policy * log_probs).sum()
    else:
        policy_loss = torch.tensor(0.0, device=device)
    return value_loss + reward_loss + policy_loss


def update_weights(
    network: MuZeroNetwork,
    optimizer: torch.optim.Optimizer,
    batch: List[BatchItem],
    device: torch.device,
) -> float:
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    for item in batch:
        observation = item.observation.unsqueeze(0).to(device)
        output = network.initial_inference(observation)
        loss = loss_for_step(output, item.targets[0], device)
        hidden_state = output.hidden_state
        for action, target in zip(item.actions, item.targets[1:]):
            action_tensor = torch.tensor([action], device=device, dtype=torch.long)
            output = network.recurrent_inference(hidden_state, action_tensor)
            hidden_state = output.hidden_state
            loss = loss + loss_for_step(output, target, device)
        total_loss = total_loss + loss
    total_loss = total_loss / max(len(batch), 1)
    total_loss.backward()
    optimizer.step()
    return float(total_loss.item())


def train(config: MuZeroConfig, device: torch.device) -> MuZeroNetwork:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    network = MuZeroNetwork(config).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    replay_buffer = ReplayBuffer(config)

    for step in range(config.training_steps):
        game = play_game(config, network, device)
        replay_buffer.save_game(game)
        if len(replay_buffer) < config.min_replay_size:
            continue
        for _ in range(config.training_steps_per_game):
            batch = replay_buffer.sample_batch()
            loss = update_weights(network, optimizer, batch, device)
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}: replay={len(replay_buffer)} loss={loss:.4f}")

    return network


def play_against_random(network: MuZeroNetwork, config: MuZeroConfig, device: torch.device) -> None:
    env = TicTacToe()
    num_moves = 0
    while not env.is_terminal() and num_moves < 9:
        if env.to_play == 1:
            root = Node(0.0)
            with torch.no_grad():
                network_output = network.initial_inference(env.make_observation().unsqueeze(0).to(device))
            expand_node(root, env.to_play, env.legal_actions(), network_output)
            run_mcts(config, root, env, network, device)
            action = select_action(config, num_moves, root)
        else:
            action = random.choice(env.legal_actions())
        env.apply(action)
        num_moves += 1
    print(env.render())
    winner = env.check_winner()
    if winner == 0:
        print("Draw")
    else:
        print("Winner:", "X" if winner == 1 else "O")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuZero Tic Tac Toe (PyTorch)")
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MuZeroConfig(training_steps=args.training_steps, num_simulations=args.num_simulations)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = MuZeroNetwork(config).to(device)
    if args.train:
        network = train(config, device)
    if args.play:
        play_against_random(network, config, device)


if __name__ == "__main__":
    main()
