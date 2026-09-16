"""MiniCard v1: the Stage 0 environment (spec section P2).

A deliberately small alternating-turn card game with hidden information. It is
*not* Hearthstone and does not wrap any Hearthstone simulator: the point is an
environment whose rules are fully known, fast, and independently testable, so
that a defect here can never be mistaken for a topology effect later.

Keywords (taunt, charge, deathrattle) are deliberately absent in v1 and are
reserved as the held-out rule variation for Stage 4. Observability is a
parameter of one engine rather than two engines, so Stage 2 (full) and Stage 3
(opponent hand hidden) cannot silently diverge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum

MAX_BOARD = 5
MAX_HAND = 7
MAX_MANA = 10
#: Amendment A2 (2026-09-16) raised this from P2's original 20. At 20 the
#: degenerate racing strategy dominated board control 88% and games ended by
#: turn 5-6, leaving nothing worth learning and no demand for working memory.
#: At 30 the strategic policy wins instead and games run 12-18 turns.
START_HP = 30
DECK_SIZE = 30
DEFAULT_MAX_TURNS = 30


class IllegalAction(ValueError):
    """Raised when a caller submits an action outside ``legal_actions()``."""


class ActionKind(IntEnum):
    END_TURN = 0
    PLAY_CARD = 1
    ATTACK = 2


class TargetKind(IntEnum):
    NONE = 0
    ENEMY_FACE = 1
    ENEMY_MINION = 2
    OWN_MINION = 3


@dataclass(frozen=True)
class Action:
    """One decision, in the three levels P2 specifies.

    ``kind`` is level 1, ``source`` is level 2 (a hand or board slot) and the
    ``target_*`` pair is level 3.
    """

    kind: ActionKind
    source: int = -1
    target_kind: TargetKind = TargetKind.NONE
    target_index: int = -1


def action(
    kind: ActionKind,
    source: int = -1,
    target_kind: TargetKind = TargetKind.NONE,
    target_index: int = -1,
) -> Action:
    return Action(kind, source, target_kind, target_index)


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    attack: int = 0
    health: int = 0
    effect: str = ""  # "" for minions

    @property
    def is_minion(self) -> bool:
        return not self.effect

    @property
    def needs_target(self) -> bool:
        return self.effect == "damage"


#: Six minions on a cost 1-5 curve and four spells, as fixed by P2.
CARD_POOL: tuple[Card, ...] = (
    Card("Wisp", cost=1, attack=1, health=1),
    Card("Scout", cost=2, attack=2, health=3),
    Card("Brute", cost=3, attack=4, health=2),
    Card("Soldier", cost=3, attack=3, health=4),
    Card("Knight", cost=4, attack=4, health=5),
    Card("Golem", cost=5, attack=5, health=7),
    Card("Bolt", cost=2, effect="damage"),
    Card("Wave", cost=3, effect="sweep"),
    Card("Mend", cost=2, effect="heal"),
    Card("Insight", cost=2, effect="draw"),
)

_BY_NAME = {card.name: card for card in CARD_POOL}

#: Deck recipes; counts sum to DECK_SIZE. Training uses "aggro" and "control";
#: "tempo" and "burn" are reserved as held-out archetypes for Stage 4.
ARCHETYPES: dict[str, dict[str, int]] = {
    "aggro": {
        "Wisp": 6, "Scout": 6, "Brute": 6, "Soldier": 2, "Knight": 2,
        "Golem": 0, "Bolt": 4, "Wave": 0, "Mend": 0, "Insight": 4,
    },
    "control": {
        "Wisp": 0, "Scout": 4, "Brute": 0, "Soldier": 6, "Knight": 6,
        "Golem": 4, "Bolt": 2, "Wave": 4, "Mend": 2, "Insight": 2,
    },
    "tempo": {
        "Wisp": 2, "Scout": 6, "Brute": 4, "Soldier": 6, "Knight": 4,
        "Golem": 2, "Bolt": 4, "Wave": 0, "Mend": 0, "Insight": 2,
    },
    "burn": {
        "Wisp": 4, "Scout": 4, "Brute": 6, "Soldier": 0, "Knight": 2,
        "Golem": 0, "Bolt": 6, "Wave": 2, "Mend": 0, "Insight": 6,
    },
}


@dataclass
class Minion:
    name: str
    attack: int
    hp: int
    sick: bool = True
    attacks_used: int = 0

    @property
    def can_attack(self) -> bool:
        return not self.sick and self.attacks_used == 0 and self.attack > 0


@dataclass
class PlayerState:
    hp: int = START_HP
    mana: int = 0
    max_mana: int = 0
    fatigue: int = 0
    deck: list[Card] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    board: list[Minion] = field(default_factory=list)


class MiniCard:
    """Two-player alternating card game with a legality-masked action space."""

    def __init__(
        self,
        seed: int = 0,
        archetypes: tuple[str, str] = ("aggro", "control"),
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.seed = seed
        self.archetypes = archetypes
        self.max_turns = max_turns
        self.players: list[PlayerState] = []
        self.to_move = 0
        self.turn = 1
        self.is_over = False
        self._winner: int | None = None

    # ---------------------------------------------------------------- setup

    def reset(self) -> None:
        rng = random.Random(self.seed)
        self.players = [PlayerState(), PlayerState()]
        self.to_move = 0
        self.turn = 1
        self.is_over = False
        self._winner = None

        for index, player in enumerate(self.players):
            player.deck = self._build_deck(self.archetypes[index], rng)
            # The second player draws one extra card instead of a coin.
            for _ in range(3 + index):
                player.hand.append(player.deck.pop())

        self._begin_turn()

    @staticmethod
    def _build_deck(archetype: str, rng: random.Random) -> list[Card]:
        recipe = ARCHETYPES[archetype]
        deck = [_BY_NAME[name] for name, count in recipe.items() for _ in range(count)]
        if len(deck) != DECK_SIZE:
            raise ValueError(f"archetype {archetype!r} has {len(deck)} cards")
        rng.shuffle(deck)
        return deck

    # ------------------------------------------------------------ accessors

    @property
    def current(self) -> PlayerState:
        return self.players[self.to_move]

    @property
    def opponent(self) -> PlayerState:
        return self.players[1 - self.to_move]

    def card_in_hand(self, player: PlayerState, index: int) -> Card:
        return player.hand[index]

    def result(self) -> int:
        """Terminal return from player 0's perspective: +1 / -1 / 0."""
        if self._winner is None:
            return 0
        return 1 if self._winner == 0 else -1

    # -------------------------------------------------------------- legality

    def legal_actions(self) -> list[Action]:
        if self.is_over:
            return []

        me, foe = self.current, self.opponent
        options = [action(ActionKind.END_TURN)]

        for index, card in enumerate(me.hand):
            if card.cost > me.mana:
                continue
            if card.is_minion:
                if len(me.board) < MAX_BOARD:
                    options.append(action(ActionKind.PLAY_CARD, source=index))
            elif card.needs_target:
                options.append(
                    action(
                        ActionKind.PLAY_CARD,
                        source=index,
                        target_kind=TargetKind.ENEMY_FACE,
                    )
                )
                options.extend(
                    action(
                        ActionKind.PLAY_CARD,
                        source=index,
                        target_kind=TargetKind.ENEMY_MINION,
                        target_index=slot,
                    )
                    for slot in range(len(foe.board))
                )
            else:
                options.append(action(ActionKind.PLAY_CARD, source=index))

        for slot, minion in enumerate(me.board):
            if not minion.can_attack:
                continue
            options.append(
                action(
                    ActionKind.ATTACK, source=slot, target_kind=TargetKind.ENEMY_FACE
                )
            )
            options.extend(
                action(
                    ActionKind.ATTACK,
                    source=slot,
                    target_kind=TargetKind.ENEMY_MINION,
                    target_index=target,
                )
                for target in range(len(foe.board))
            )

        return options

    # ------------------------------------------------------------------ step

    def step(self, chosen: Action) -> None:
        if self.is_over:
            raise IllegalAction("the game is already over")
        if chosen not in self.legal_actions():
            raise IllegalAction(f"{chosen!r} is not legal in this state")

        if chosen.kind is ActionKind.END_TURN:
            self._end_turn()
            return
        if chosen.kind is ActionKind.PLAY_CARD:
            self._play_card(chosen)
        else:
            self._attack(chosen)
        self._check_terminal()

    def _play_card(self, chosen: Action) -> None:
        me, foe = self.current, self.opponent
        card = me.hand.pop(chosen.source)
        me.mana -= card.cost

        if card.is_minion:
            me.board.append(Minion(card.name, card.attack, card.health))
        elif card.effect == "damage":
            if chosen.target_kind is TargetKind.ENEMY_FACE:
                foe.hp -= 3
            else:
                self._damage_minion(foe, chosen.target_index, 3)
        elif card.effect == "sweep":
            for slot in reversed(range(len(foe.board))):
                self._damage_minion(foe, slot, 2)
        elif card.effect == "heal":
            me.hp = min(START_HP, me.hp + 5)
        elif card.effect == "draw":
            self._draw(me, 2)

    def _attack(self, chosen: Action) -> None:
        me, foe = self.current, self.opponent
        attacker = me.board[chosen.source]
        attacker.attacks_used += 1

        if chosen.target_kind is TargetKind.ENEMY_FACE:
            foe.hp -= attacker.attack
            return

        defender = foe.board[chosen.target_index]
        attacker.hp -= defender.attack
        self._damage_minion(foe, chosen.target_index, attacker.attack)
        if attacker.hp <= 0:
            me.board.remove(attacker)

    @staticmethod
    def _damage_minion(owner: PlayerState, slot: int, amount: int) -> None:
        minion = owner.board[slot]
        minion.hp -= amount
        if minion.hp <= 0:
            owner.board.pop(slot)

    # ------------------------------------------------------------ turn cycle

    def _end_turn(self) -> None:
        self.to_move = 1 - self.to_move
        if self.to_move == 0:
            self.turn += 1
        if self.turn > self.max_turns:
            self.is_over = True
            self._winner = None
            return
        self._begin_turn()
        self._check_terminal()

    def _begin_turn(self) -> None:
        player = self.current
        player.max_mana = min(self.turn, MAX_MANA)
        player.mana = player.max_mana
        for minion in player.board:
            minion.sick = False
            minion.attacks_used = 0
        self._draw(player, 1)

    def _draw(self, player: PlayerState, count: int) -> None:
        for _ in range(count):
            if not player.deck:
                player.fatigue += 1
                player.hp -= player.fatigue
                continue
            card = player.deck.pop()
            if len(player.hand) < MAX_HAND:
                player.hand.append(card)
            # Otherwise the card is drawn and burned, as in the genre.

    def _check_terminal(self) -> None:
        dead = [index for index, player in enumerate(self.players) if player.hp <= 0]
        if not dead:
            return
        self.is_over = True
        self._winner = None if len(dead) == 2 else 1 - dead[0]

    # ---------------------------------------------------------- observation

    def observe(self, player_index: int, hide_opponent_hand: bool = True) -> dict:
        """Snapshot from one player's seat.

        Deck *order* is never exposed in either stage, only its size. Setting
        ``hide_opponent_hand`` is the only difference between Stage 2 and
        Stage 3.
        """
        me = self.players[player_index]
        foe = self.players[1 - player_index]

        def side(player: PlayerState, hand_visible: bool) -> dict:
            return {
                "hp": player.hp,
                "mana": player.mana,
                "max_mana": player.max_mana,
                "deck_size": len(player.deck),
                "hand_size": len(player.hand),
                "hand": [card.name for card in player.hand] if hand_visible else None,
                "board": [
                    {"name": m.name, "attack": m.attack, "hp": m.hp, "sick": m.sick}
                    for m in player.board
                ],
            }

        return {
            "turn": self.turn,
            "to_move": self.to_move,
            "own": side(me, True),
            "opponent": side(foe, not hide_opponent_hand),
        }

    # ------------------------------------------------------------- test hook

    def summon_for_test(self, player: PlayerState, card_name: str) -> Minion:
        """Place a minion directly on the board, bypassing hand and mana."""
        card = _BY_NAME[card_name]
        minion = Minion(card.name, card.attack, card.health)
        player.board.append(minion)
        return minion


# ----------------------------------------------------------- scripted play


def greedy_face_opponent(game: MiniCard) -> Action:
    """Develop the board, then send everything at the opponent's face."""
    options = game.legal_actions()

    face = [
        a
        for a in options
        if a.kind is ActionKind.ATTACK and a.target_kind is TargetKind.ENEMY_FACE
    ]
    if face:
        return face[0]

    plays = [a for a in options if a.kind is ActionKind.PLAY_CARD]
    if plays:
        return max(plays, key=lambda a: game.card_in_hand(game.current, a.source).cost)

    return action(ActionKind.END_TURN)


def board_control_opponent(game: MiniCard) -> Action:
    """Trade into enemy minions first; only swing at the face once it is clear."""
    options = game.legal_actions()

    trades = [
        a
        for a in options
        if a.kind is ActionKind.ATTACK and a.target_kind is TargetKind.ENEMY_MINION
    ]
    if trades:
        return max(trades, key=lambda a: game.opponent.board[a.target_index].attack)

    plays = [a for a in options if a.kind is ActionKind.PLAY_CARD]
    if plays:
        return max(plays, key=lambda a: game.card_in_hand(game.current, a.source).cost)

    face = [
        a
        for a in options
        if a.kind is ActionKind.ATTACK and a.target_kind is TargetKind.ENEMY_FACE
    ]
    if face:
        return face[0]

    return action(ActionKind.END_TURN)
