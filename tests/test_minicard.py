"""Tests for the MiniCard v1 engine (spec section P2).

Stage 0 of the curriculum. The engine must be deterministic and correct on its
own terms before any neural substrate is attached to it, because a bug here
would be indistinguishable from a topology effect later.
"""

import pytest

from flywire_rl.minicard import (
    ActionKind,
    IllegalAction,
    MiniCard,
    TargetKind,
    action,
    greedy_face_opponent,
)


def _game(seed=0, **kwargs):
    game = MiniCard(seed=seed, **kwargs)
    game.reset()
    return game


def test_same_seed_reproduces_an_identical_game():
    def play(seed):
        game = _game(seed)
        trace = []
        while not game.is_over:
            chosen = greedy_face_opponent(game)
            trace.append(chosen)
            game.step(chosen)
        return trace, game.result()

    assert play(7) == play(7)


def test_different_seeds_diverge():
    # Guards against a "deterministic" engine that ignores its seed entirely.
    def final(seed):
        game = _game(seed)
        while not game.is_over:
            game.step(greedy_face_opponent(game))
        return game.turn, game.players[0].hp, game.players[1].hp

    assert len({final(s) for s in range(6)}) > 1


def test_mana_follows_the_turn_index_and_caps_at_ten():
    game = _game()
    seen = []
    while not game.is_over and game.turn <= 12:
        seen.append((game.turn, game.current.max_mana))
        game.step(action(ActionKind.END_TURN))

    for turn, max_mana in seen:
        assert max_mana == min(turn, 10)


def test_mana_refreshes_each_turn():
    game = _game()
    game.current.mana = 0
    game.step(action(ActionKind.END_TURN))
    game.step(action(ActionKind.END_TURN))

    assert game.current.mana == game.current.max_mana


def test_end_turn_is_always_legal():
    game = _game()
    assert action(ActionKind.END_TURN) in game.legal_actions()


def test_illegal_action_raises_rather_than_being_ignored():
    game = _game()
    # Hand slot 6 is empty on turn 1, so playing it is not a legal action.
    with pytest.raises(IllegalAction):
        game.step(action(ActionKind.PLAY_CARD, source=6))


def test_cards_cost_mana_and_unaffordable_cards_are_illegal():
    game = _game()
    game.current.mana = 0

    plays = [a for a in game.legal_actions() if a.kind is ActionKind.PLAY_CARD]
    assert plays == []


def test_board_is_capped_at_five_minions():
    game = _game()
    player = game.current
    for _ in range(5):
        game.summon_for_test(player, "Wisp")

    assert len(player.board) == 5
    plays = [
        a
        for a in game.legal_actions()
        if a.kind is ActionKind.PLAY_CARD
        and game.card_in_hand(player, a.source).is_minion
    ]
    assert plays == []


def test_summoned_minions_cannot_attack_on_the_turn_they_arrive():
    game = _game()
    minion = game.summon_for_test(game.current, "Scout")

    assert minion.sick
    attacks = [a for a in game.legal_actions() if a.kind is ActionKind.ATTACK]
    assert attacks == []


def test_a_minion_may_attack_once_per_turn():
    game = _game()
    minion = game.summon_for_test(game.current, "Scout")
    minion.sick = False

    strike = action(ActionKind.ATTACK, source=0, target_kind=TargetKind.ENEMY_FACE)
    assert strike in game.legal_actions()
    game.step(strike)

    assert minion.attacks_used == 1
    assert strike not in game.legal_actions()


def test_attacking_the_face_reduces_opponent_hp_by_attack():
    game = _game()
    minion = game.summon_for_test(game.current, "Knight")  # 4 attack
    minion.sick = False
    defender = game.opponent
    before = defender.hp

    game.step(action(ActionKind.ATTACK, source=0, target_kind=TargetKind.ENEMY_FACE))

    assert defender.hp == before - 4


def test_minions_trade_damage_when_they_fight():
    game = _game()
    attacker = game.summon_for_test(game.current, "Knight")  # 4/5
    attacker.sick = False
    defender = game.summon_for_test(game.opponent, "Scout")  # 2/3

    game.step(
        action(
            ActionKind.ATTACK,
            source=0,
            target_kind=TargetKind.ENEMY_MINION,
            target_index=0,
        )
    )

    assert defender not in game.opponent.board  # 3 health, took 4
    assert attacker.hp == 5 - 2


def test_fatigue_damages_a_player_whose_deck_is_empty():
    game = _game()
    player = game.current
    player.deck.clear()
    before = player.hp

    game.step(action(ActionKind.END_TURN))
    game.step(action(ActionKind.END_TURN))  # back to the same player

    assert player.hp < before


def test_hand_is_capped_and_overdraw_is_discarded():
    game = _game()
    player = game.current
    while len(player.hand) < 7:
        player.hand.append(player.deck.pop())
    deck_before = len(player.deck)

    game.step(action(ActionKind.END_TURN))
    game.step(action(ActionKind.END_TURN))

    assert len(player.hand) == 7
    assert len(player.deck) < deck_before  # the card was drawn, then burned


def test_game_ends_when_a_player_reaches_zero_hp():
    game = _game()
    game.opponent.hp = 1
    minion = game.summon_for_test(game.current, "Knight")
    minion.sick = False

    game.step(action(ActionKind.ATTACK, source=0, target_kind=TargetKind.ENEMY_FACE))

    assert game.is_over
    assert game.result() == 1  # player 0 to move won


def test_turn_limit_ends_the_game_as_a_draw():
    game = _game(max_turns=3)
    while not game.is_over:
        game.step(action(ActionKind.END_TURN))

    assert game.is_over
    assert game.result() == 0


def test_stage_two_reveals_the_opponent_hand_and_stage_three_hides_it():
    game = _game()
    game.step(action(ActionKind.END_TURN))  # give the opponent a hand

    visible = game.observe(0, hide_opponent_hand=False)
    hidden = game.observe(0, hide_opponent_hand=True)

    assert visible["opponent"]["hand"] is not None
    assert hidden["opponent"]["hand"] is None
    assert hidden["opponent"]["hand_size"] == len(game.players[1].hand)


def test_deck_order_is_never_observable():
    game = _game()
    for hide in (False, True):
        observation = game.observe(0, hide_opponent_hand=hide)
        assert "deck" not in observation["own"]
        assert "deck" not in observation["opponent"]
        assert observation["own"]["deck_size"] == len(game.players[0].deck)


def test_legal_actions_are_never_empty_while_the_game_runs():
    game = _game()
    steps = 0
    while not game.is_over and steps < 400:
        options = game.legal_actions()
        assert options
        game.step(greedy_face_opponent(game))
        steps += 1


def test_archetypes_produce_different_decks():
    a = MiniCard(seed=0, archetypes=("aggro", "aggro"))
    b = MiniCard(seed=0, archetypes=("control", "control"))
    a.reset()
    b.reset()

    assert [c.name for c in a.players[0].deck] != [c.name for c in b.players[0].deck]


def test_every_deck_has_thirty_cards():
    game = _game()
    for player in game.players:
        assert len(player.deck) + len(player.hand) == 30
