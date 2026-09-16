"""Observation encoder, hierarchical action decoder, and the spiking policy.

The joint between MiniCard (P2) and the LIF substrate (P3).

Two design commitments carry most of the weight here.

**Legality is structural, not learned.** Each of the three action levels is
sampled from logits masked to exactly the legal set, conditioned on the levels
already chosen. A sampled action is therefore always legal by construction, so
the invalid-action rate is zero rather than something the agent must be
penalised into. Nothing in the reward has to pay for rule-following.

**The hidden-information boundary is explicit.** When the opponent's hand is
concealed its one-hot block is zeroed *and* a visibility flag is cleared, so the
network can tell "hidden" from "genuinely empty". Without the flag those two
states are the same vector and Stage 3 would quietly corrupt its own evidence.

Per P3 the input and output neuron *indices* are supplied by the caller and are
identical across arms: arms differ in their edges, never in which nodes the game
is wired to, nor in trainable capacity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from flywire_rl.minicard import (
    CARD_POOL,
    MAX_BOARD,
    MAX_HAND,
    MAX_MANA,
    START_HP,
    Action,
    ActionKind,
    TargetKind,
    action,
)
from flywire_rl.spiking import (
    CouplingMode,
    LIFSubstrate,
    LowRankCoupling,
    ShiuParams,
)

_CARD_INDEX = {card.name: i for i, card in enumerate(CARD_POOL)}
N_CARD_TYPES = len(CARD_POOL)

_SCALARS_PER_SIDE = 5  # hp, mana, max_mana, deck_size, hand_size
_MINION_FEATURES = N_CARD_TYPES + 3  # one-hot + attack, hp, sick


def _side_width() -> int:
    return (
        _SCALARS_PER_SIDE
        + MAX_HAND * N_CARD_TYPES
        + MAX_BOARD * _MINION_FEATURES
    )


#: turn + own side + opponent side + hand-visibility flag
OBS_DIM = 1 + 2 * _side_width() + 1


class ActionCodec:
    """Flat integer codes for the three action levels, plus legality masks."""

    N_KIND = len(ActionKind)
    N_SOURCE = MAX_HAND + MAX_BOARD  # hand slots 0-6, board slots 7-11
    N_TARGET = 2 + 2 * MAX_BOARD  # none, enemy face, 5 enemy, 5 own

    NO_SLOT = N_SOURCE  # sentinel for actions that take no source

    # Offsets inside the encoded observation, used by tests and by analysis.
    _OWN_OFFSET = 1
    _OPP_OFFSET = 1 + _side_width()
    OPPONENT_HAND_SIZE_INDEX = _OPP_OFFSET + 4
    OPPONENT_HAND_SLICE = (
        _OPP_OFFSET + _SCALARS_PER_SIDE,
        _OPP_OFFSET + _SCALARS_PER_SIDE + MAX_HAND * N_CARD_TYPES,
    )
    HAND_VISIBLE_INDEX = OBS_DIM - 1

    @staticmethod
    def encode(chosen: Action) -> tuple[int, int, int]:
        if chosen.kind is ActionKind.END_TURN:
            return int(ActionKind.END_TURN), ActionCodec.NO_SLOT, 0

        source = chosen.source
        if chosen.kind is ActionKind.ATTACK:
            source += MAX_HAND

        if chosen.target_kind is TargetKind.NONE:
            target = 0
        elif chosen.target_kind is TargetKind.ENEMY_FACE:
            target = 1
        elif chosen.target_kind is TargetKind.ENEMY_MINION:
            target = 2 + chosen.target_index
        else:
            target = 2 + MAX_BOARD + chosen.target_index

        return int(chosen.kind), source, target

    @staticmethod
    def decode(kind_code: int, source_code: int, target_code: int) -> Action:
        kind = ActionKind(kind_code)
        if kind is ActionKind.END_TURN:
            return action(ActionKind.END_TURN)

        source = source_code - MAX_HAND if kind is ActionKind.ATTACK else source_code

        if target_code == 0:
            return action(kind, source)
        if target_code == 1:
            return action(kind, source, TargetKind.ENEMY_FACE)
        if target_code < 2 + MAX_BOARD:
            return action(kind, source, TargetKind.ENEMY_MINION, target_code - 2)
        return action(kind, source, TargetKind.OWN_MINION, target_code - 2 - MAX_BOARD)

    @staticmethod
    def kind_mask(legal: list[Action]) -> np.ndarray:
        mask = np.zeros(ActionCodec.N_KIND, dtype=bool)
        for candidate in legal:
            mask[int(candidate.kind)] = True
        return mask

    @staticmethod
    def source_mask(legal: list[Action], kind: ActionKind) -> np.ndarray:
        mask = np.zeros(ActionCodec.N_SOURCE + 1, dtype=bool)
        for candidate in legal:
            if candidate.kind is kind:
                mask[ActionCodec.encode(candidate)[1]] = True
        return mask

    @staticmethod
    def target_mask(legal: list[Action], kind: ActionKind, source: int) -> np.ndarray:
        mask = np.zeros(ActionCodec.N_TARGET, dtype=bool)
        for candidate in legal:
            code = ActionCodec.encode(candidate)
            if candidate.kind is kind and code[1] == source:
                mask[code[2]] = True
        return mask


def _encode_side(out: np.ndarray, offset: int, side: dict) -> int:
    out[offset + 0] = side["hp"] / START_HP
    out[offset + 1] = side["mana"] / MAX_MANA
    out[offset + 2] = side["max_mana"] / MAX_MANA
    out[offset + 3] = side["deck_size"] / 30.0
    out[offset + 4] = side["hand_size"] / MAX_HAND
    cursor = offset + _SCALARS_PER_SIDE

    for slot in range(MAX_HAND):
        names = side["hand"]
        if names is not None and slot < len(names):
            out[cursor + _CARD_INDEX[names[slot]]] = 1.0
        cursor += N_CARD_TYPES

    for slot in range(MAX_BOARD):
        if slot < len(side["board"]):
            minion = side["board"][slot]
            out[cursor + _CARD_INDEX[minion["name"]]] = 1.0
            out[cursor + N_CARD_TYPES + 0] = minion["attack"] / 10.0
            out[cursor + N_CARD_TYPES + 1] = minion["hp"] / 10.0
            out[cursor + N_CARD_TYPES + 2] = 1.0 if minion["sick"] else 0.0
        cursor += _MINION_FEATURES

    return cursor


def encode_observation(observation: dict) -> np.ndarray:
    """Flatten an ``observe()`` snapshot into a fixed-width, bounded vector."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    out[0] = observation["turn"] / 30.0

    cursor = _encode_side(out, ActionCodec._OWN_OFFSET, observation["own"])
    _encode_side(out, cursor, observation["opponent"])
    out[ActionCodec.HAND_VISIBLE_INDEX] = (
        0.0 if observation["opponent"]["hand"] is None else 1.0
    )
    return out


#: Afferent classes: the game is read in where the brain reads the world in.
INPUT_CLASSES = ("sensory", "visual_projection", "ascending")
#: Efferent classes: descending neurons are the brain's actual motor output.
OUTPUT_CLASSES = ("descending", "motor")


def select_io_indices(
    super_classes, input_classes=INPUT_CLASSES, output_classes=OUTPUT_CLASSES
) -> tuple[np.ndarray, np.ndarray]:
    """Choose input and output neurons by their FlyWire super-class.

    Wiring the game through afferent and efferent populations is more
    defensible than picking indices by degree, and it keeps P3 satisfied: the
    rule runs once on the real graph and the resulting *indices* are then reused
    unchanged by every arm, because the controls shuffle edges and never nodes.
    """
    labels = np.asarray(list(super_classes), dtype=object)
    inputs = np.flatnonzero(np.isin(labels, list(input_classes)))
    outputs = np.flatnonzero(np.isin(labels, list(output_classes)))
    if inputs.size == 0 or outputs.size == 0:
        raise ValueError("super-class labels yielded no input or no output neurons")
    return inputs, outputs


def _masked_sample(logits: Tensor, mask: np.ndarray) -> tuple[int, Tensor]:
    blocked = torch.as_tensor(~mask, device=logits.device)
    masked = logits.masked_fill(blocked, float("-inf"))
    log_probs = torch.log_softmax(masked, dim=-1)
    index = int(torch.multinomial(log_probs.exp(), 1).item())
    return index, log_probs[index]


class SpikingPolicy(nn.Module):
    """Encoder -> frozen connectome with a low-rank adapter -> masked heads."""

    def __init__(
        self,
        n_neurons: int,
        pre_idx: Tensor,
        post_idx: Tensor,
        weights: Tensor,
        input_indices: Tensor,
        output_indices: Tensor,
        rank: int = 8,
        steps: int = 300,  # 30 ms at the published dt; see experiment.DECISION_WINDOW_MS
        mode: CouplingMode = CouplingMode.SPARSE,
        input_drive_mv: float = 1.0,
        params: ShiuParams | None = None,
    ) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.steps = steps
        self.input_drive_mv = input_drive_mv
        self.params = params or ShiuParams()

        self.register_buffer("input_indices", input_indices.long())
        self.register_buffer("output_indices", output_indices.long())

        self.coupling = LowRankCoupling(
            n_neurons, pre_idx, post_idx, weights, rank=rank, mode=mode
        )
        self.substrate = LIFSubstrate(self.coupling, params=self.params)

        n_in = int(input_indices.numel())
        n_out = int(output_indices.numel())
        self.encoder = nn.Linear(OBS_DIM, n_in)
        self.head_kind = nn.Linear(n_out, ActionCodec.N_KIND)
        self.head_source = nn.Linear(n_out, ActionCodec.N_SOURCE + 1)
        self.head_target = nn.Linear(n_out, ActionCodec.N_TARGET)
        self.head_value = nn.Linear(n_out, 1)

        # Amplitude control (fly-ky7.11). Frozen affine statistics applied to
        # the readout before the heads; buffers, not parameters, so they are
        # saved with the policy but never receive a gradient. ``standardised``
        # is a buffer too, so a reloaded policy cannot silently forget that it
        # was calibrated. Disabled until calibrate_readout is called, which
        # keeps the pre-amendment arms bit-identical.
        self.register_buffer("readout_mean", torch.zeros(n_out))
        self.register_buffer("readout_scale", torch.ones(n_out))
        self.register_buffer("standardised", torch.zeros((), dtype=torch.bool))

    @property
    def device(self) -> torch.device:
        return self.encoder.weight.device

    def raw_features(self, observation: Tensor) -> Tensor:
        """Run the substrate and return each output neuron's mean membrane, in mV.

        Amendment A4 reached for the membrane because a spike-count readout left
        the median output unit with zero variance across game states -- a
        symptom of the branching calibration, not of the readout. With the
        absolute scale in place the choice is no longer forced, but the membrane
        is kept: it is graded, so it carries sub-threshold evidence that a count
        discards, and it never collapses to a constant. Note the resting value
        is ``v_rest = -52 mV``, not zero, so these features are negative.
        """
        observation = observation.to(self.device)
        batch = observation.shape[0]
        currents = self.encoder(observation) * self.input_drive_mv

        drive = torch.zeros(
            batch, self.n_neurons, dtype=currents.dtype, device=currents.device
        )
        drive = drive.index_copy(1, self.input_indices, currents)
        # A sustained stimulus: the same current is presented for the whole
        # decision window rather than as a single impulse.
        result = self.substrate.rollout(drive.unsqueeze(0).expand(self.steps, -1, -1))

        return result.membrane.index_select(1, self.output_indices)

    def features(self, observation: Tensor) -> Tensor:
        """The readout the heads actually consume.

        Identical to :meth:`raw_features` until :meth:`calibrate_readout` has
        run; afterwards each output unit is shifted and scaled by the frozen
        statistics measured there. See that method for why this exists.
        """
        features = self.raw_features(observation)
        if bool(self.standardised):
            features = (features - self.readout_mean) / self.readout_scale
        return features

    @torch.no_grad()
    def calibrate_readout(
        self,
        reference: Tensor,
        eps: float = 1e-6,
        chunk: int = 32,
    ) -> dict[str, float]:
        """Measure per-unit readout statistics on a fixed reference set (A4/fly-ky7.11).

        The pilot's arm ordering (real > shuffled > random) reproduced the
        *pre-learning* ordering of readout amplitude exactly -- median s.d.
        70.1 / 13.7 / 4.88 and feature RMS 1510 / 839 / 242 in amendment A4. So
        the pilot may be reporting "the real connectome delivers more signal to
        the readout", not "the real connectome computes better". Those are
        different claims and no P4 control separates them.

        Standardising each arm's readout to zero mean and unit variance removes
        the amplitude difference while leaving the *pattern* across units
        intact. Re-running then answers the question: if the ordering survives
        it is computational, and if it vanishes it was transmission.

        The reference set must be the same states for every arm -- see
        :func:`reference_observations`, which is policy-independent for exactly
        that reason. Each arm then gets its own statistics measured on those
        shared states.

        The statistics are computed once and frozen. A running or per-batch
        normaliser would keep adapting during training and so would smuggle a
        second, arm-dependent learning signal into the comparison it is meant
        to neutralise.
        """
        if reference.ndim != 2:
            raise ValueError(
                f"reference must be (n_states, obs_dim), got shape {tuple(reference.shape)}"
            )
        if reference.shape[0] < 2:
            raise ValueError(
                "a variance needs at least 2 reference states, "
                f"got {reference.shape[0]}"
            )

        was = bool(self.standardised)
        self.standardised.fill_(False)  # measure the raw readout, not a standardised one
        try:
            batches = [
                self.raw_features(reference[i : i + chunk])
                for i in range(0, reference.shape[0], chunk)
            ]
            features = torch.cat(batches, dim=0)
        except BaseException:
            self.standardised.fill_(was)
            raise

        # Accumulate in float64. The membrane sits around v_rest = -52 mV, so a
        # float32 mean of values near -52 loses most of its significant digits
        # to cancellation, and a unit with a small s.d. then has that error
        # divided by the small scale. The statistics are computed once, so the
        # wider accumulator is free.
        mean = features.double().mean(dim=0).to(features.dtype)
        std = features.double().std(dim=0, unbiased=False).to(features.dtype)

        # A dead unit -- constant across every reference state -- has no scale
        # to divide by. Leaving it at 1.0 passes its (now zero) deviation
        # through untouched instead of amplifying float noise by 1/eps.
        dead = std <= eps
        scale = torch.where(dead, torch.ones_like(std), std)

        self.readout_mean.copy_(mean)
        self.readout_scale.copy_(scale)
        self.standardised.fill_(True)

        return {
            "n_states": float(reference.shape[0]),
            "n_units": float(mean.numel()),
            "dead_units": float(int(dead.sum())),
            "mean_abs_mean": float(mean.abs().mean()),
            "median_sd": float(std.median()),
            "feature_rms": float(features.pow(2).mean().sqrt()),
        }

    def act(self, game) -> tuple[Action, Tensor, Tensor]:
        """Sample one legal action, returning it with its log-prob and value."""
        legal = game.legal_actions()
        observation = encode_observation(
            game.observe(game.to_move, hide_opponent_hand=True)
        )
        features = self.features(torch.from_numpy(observation).unsqueeze(0))

        kind_code, log_kind = _masked_sample(
            self.head_kind(features)[0], ActionCodec.kind_mask(legal)
        )
        kind = ActionKind(kind_code)

        source_code, log_source = _masked_sample(
            self.head_source(features)[0], ActionCodec.source_mask(legal, kind)
        )
        target_code, log_target = _masked_sample(
            self.head_target(features)[0],
            ActionCodec.target_mask(legal, kind, source_code),
        )

        chosen = ActionCodec.decode(kind_code, source_code, target_code)
        value = self.head_value(features)[0, 0]
        return chosen, log_kind + log_source + log_target, value

    def act_with_record(self, game) -> tuple[Action, dict]:
        """Sample an action and return everything PPO needs to replay it.

        The masks are stored, not recomputed later from the game state: the
        state has moved on by update time, and a mask that no longer matches the
        one used at sampling silently corrupts the importance ratio.
        """
        legal = game.legal_actions()
        observation = encode_observation(
            game.observe(game.to_move, hide_opponent_hand=True)
        )

        # Rollout collection needs numbers, not a graph. Building one per
        # decision and discarding it costs time and memory for nothing; the
        # update recomputes everything it differentiates through.
        with torch.no_grad():
            features = self.features(torch.from_numpy(observation).unsqueeze(0))

            kind_mask = ActionCodec.kind_mask(legal)
            kind_code, log_kind = _masked_sample(self.head_kind(features)[0], kind_mask)
            kind = ActionKind(kind_code)

            source_mask = ActionCodec.source_mask(legal, kind)
            source_code, log_source = _masked_sample(
                self.head_source(features)[0], source_mask
            )

            target_mask = ActionCodec.target_mask(legal, kind, source_code)
            target_code, log_target = _masked_sample(
                self.head_target(features)[0], target_mask
            )
            value = self.head_value(features)[0, 0]

        record = {
            "observation": observation,
            "kind_code": kind_code,
            "source_code": source_code,
            "target_code": target_code,
            "kind_mask": kind_mask,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "log_prob": float(log_kind + log_source + log_target),
            "value": float(value),
        }
        return ActionCodec.decode(kind_code, source_code, target_code), record

    def evaluate_actions(
        self,
        observations: Tensor,
        kind_codes: Tensor,
        source_codes: Tensor,
        target_codes: Tensor,
        kind_masks: Tensor,
        source_masks: Tensor,
        target_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Replay stored actions in a batch, returning log-prob, value, entropy."""
        features = self.features(observations)
        device = features.device

        def level(head: nn.Module, codes: Tensor, masks: Tensor):
            logits = head(features).masked_fill(
                ~masks.to(device), torch.finfo(features.dtype).min
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            chosen = log_probs.gather(1, codes.to(device).unsqueeze(1)).squeeze(1)
            # Masked entries contribute p * log p = 0; leaving -inf in place
            # would make the product NaN rather than zero.
            safe = log_probs.masked_fill(~masks.to(device), 0.0)
            entropy = -(safe.exp() * safe).sum(-1)
            return chosen, entropy

        log_kind, ent_kind = level(self.head_kind, kind_codes, kind_masks)
        log_source, ent_source = level(self.head_source, source_codes, source_masks)
        log_target, ent_target = level(self.head_target, target_codes, target_masks)

        values = self.head_value(features).squeeze(-1)
        return (
            log_kind + log_source + log_target,
            values,
            ent_kind + ent_source + ent_target,
        )


def reference_observations(
    n_states: int = 64,
    seed: int = 20260916,
    archetype: str = "aggro",
    max_turns: int = 30,
) -> Tensor:
    """A fixed set of game states for calibrating the readout (fly-ky7.11).

    Deliberately **policy-independent**: states are reached by sampling uniformly
    from the legal set with a dedicated RNG, never by asking a network what to
    do. If the reference states were drawn from each arm's own behaviour the
    arms would be standardised on different distributions, and the amplitude
    control would compare readouts that had been centred on different regions
    of the state space -- reintroducing the very confound it exists to remove.

    Positions are collected one per game across ``n_states`` distinct seeds, at
    a random legal depth, so the set spans openings, midgame and near-terminal
    boards rather than ``n_states`` correlated steps of a single match.
    """
    import random

    from flywire_rl.minicard import MiniCard

    if n_states < 2:
        raise ValueError(f"a variance needs at least 2 reference states, got {n_states}")

    rng = random.Random(seed)
    rows: list[np.ndarray] = []

    for i in range(n_states):
        game = MiniCard(
            seed=seed + 7919 * i, archetypes=(archetype, archetype), max_turns=max_turns
        )
        game.reset()
        for _ in range(rng.randrange(0, 24)):
            if game.is_over:
                break
            legal = game.legal_actions()
            if not legal:
                break
            game.step(rng.choice(legal))
        rows.append(
            encode_observation(game.observe(game.to_move, hide_opponent_hand=True))
        )

    return torch.from_numpy(np.stack(rows))
