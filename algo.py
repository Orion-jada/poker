#!/usr/bin/env python3
"""
Full-featured, efficient, non-ML poker AI (Texas Hold'em)
- 7-card evaluator (bitboard-based straight detection + flush detection)
- Monte Carlo equity simulation with caching
- Rich OpponentModel (rolling stats, derived metrics)
- Decision engine with preflop heuristics, postflop EV, pot odds, fold equity,
  multiway adjustments, and exploitative sizing.
"""

from collections import defaultdict, deque
from functools import lru_cache
import itertools
import math
import random
import time
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}  # 0..12
INT_TO_RANK = {i: r for r, i in RANK_TO_INT.items()}

# -----------------------------
# Card utilities & evaluator
# -----------------------------
def card_to_rank(card: str) -> int:
    return RANK_TO_INT[card[0]]

def card_to_suit(card: str) -> str:
    return card[1]

def card_to_bit(card: str) -> Tuple[int, int]:
    """Return (rank_bitmask, suit_index) where rank_bitmask is 1<<rank"""
    r = card_to_rank(card)
    return 1 << r, SUITS.index(card_to_suit(card))

def ranks_bitmask_from_cards(cards: List[str]) -> int:
    mask = 0
    for c in cards:
        mask |= (1 << card_to_rank(c))
    return mask

STRAIGHT_MASKS = []
# Precompute straight masks for ranks 0..12 (A-high to 5-low special)
for top in range(12, 3, -1):  # top rank index of straight (A=12 down to 4)
    m = 0
    for offset in range(5):
        m |= 1 << (top - offset)
    STRAIGHT_MASKS.append(m)
# wheel (A-2-3-4-5)
STRAIGHT_MASKS.append((1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3))

def highest_straight_rank(ranks_mask: int) -> int:
    """Return top rank index of straight if exists, else -1"""
    # check masks from highest to lowest
    for m in STRAIGHT_MASKS:
        if (ranks_mask & m) == m:
            # determine top rank index
            # wheel special case
            if m == STRAIGHT_MASKS[-1]:
                return 3  # 5 is top in wheel (index 3)
            # find highest set bit in m
            for i in range(12, -1, -1):
                if (m >> i) & 1:
                    return i
    return -1

def best_five_from_seven(cards: List[str]) -> Tuple[int, Tuple]:
    """
    Evaluate 7-card hand, return (category, tiebreaker_tuple)
    Categories: 8=StraightFlush,7=FourKind,6=FullHouse,5=Flush,4=Straight,3=Trips,2=TwoPair,1=Pair,0=HighCard
    Tiebreaker tuple ensures lexicographic comparison for equality.
    """
    assert 5 <= len(cards) <= 7
    # Group by suit, by rank
    suits = {s: [] for s in SUITS}
    rank_counts = np.zeros(13, dtype=int)  # Use NumPy for faster counting
    for c in cards:
        r = card_to_rank(c)
        rank_counts[r] += 1
        suits[card_to_suit(c)].append(c)

    # Build ranks mask
    ranks_mask = 0
    for i in range(13):
        if rank_counts[i]:
            ranks_mask |= (1 << i)

    # Check flush (if any suit has >=5 cards)
    flush_suit = None
    flush_cards = []
    for s in SUITS:
        if len(suits[s]) >= 5:
            flush_suit = s
            # sort flush cards by rank desc
            flush_cards = sorted(suits[s], key=lambda c: card_to_rank(c), reverse=True)
            break

    # Straight flush
    if flush_suit:
        flush_ranks_mask = 0
        for c in suits[flush_suit]:
            flush_ranks_mask |= 1 << card_to_rank(c)
        sf_top = highest_straight_rank(flush_ranks_mask)
        if sf_top != -1:
            return 8, (sf_top,)

    # Four of a kind
    quad_rank = next((i for i, cnt in enumerate(rank_counts[::-1]) if cnt == 4), None)
    if quad_rank is not None:
        # quad_rank is index in reversed list; convert
        quad_rank = 12 - quad_rank
        # kicker highest other card
        kicker = max(i for i in range(13) if i != quad_rank and rank_counts[i] > 0)
        return 7, (quad_rank, kicker)

    # Full house (3 + 2)
    trips = [i for i in range(12, -1, -1) if rank_counts[i] >= 3]
    pairs = [i for i in range(12, -1, -1) if rank_counts[i] >= 2]
    if trips:
        # remove the trips used for pair search
        used_trip = trips[0]
        # find highest pair that's not the trip rank (could be another trips)
        pair_for_full = next((p for p in pairs if p != used_trip), None)
        if pair_for_full is not None:
            return 6, (used_trip, pair_for_full)

    # Flush
    if flush_suit:
        top5 = tuple(card_to_rank(c) for c in flush_cards[:5])
        return 5, top5

    # Straight
    st_top = highest_straight_rank(ranks_mask)
    if st_top != -1:
        return 4, (st_top,)

    # Trips
    if trips:
        trip_rank = trips[0]
        kickers = tuple(i for i in range(12, -1, -1) if i != trip_rank and rank_counts[i] > 0)[:2]
        return 3, (trip_rank,) + tuple(kickers)

    # Two pair
    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = next(i for i in range(12, -1, -1) if i not in (high, low) and rank_counts[i] > 0)
        return 2, (high, low, kicker)

    # Pair
    if len(pairs) == 1:
        pr = pairs[0]
        kickers = tuple(i for i in range(12, -1, -1) if i != pr and rank_counts[i] > 0)[:3]
        return 1, (pr,) + kickers

    # High card
    high_cards = tuple(i for i in range(12, -1, -1) if rank_counts[i] > 0)[:5]
    return 0, high_cards

def compare_hands(a: Tuple[int, Tuple], b: Tuple[int, Tuple]) -> int:
    """Return 1 if a>b, -1 if a<b, 0 if tie. a and b are (category, tiebreaker_tuple)"""
    if a[0] != b[0]:
        return 1 if a[0] > b[0] else -1
    # compare tiebreakers lexicographically
    if a[1] > b[1]:
        return 1
    elif a[1] < b[1]:
        return -1
    return 0

# -----------------------------
# Equity Simulation (Monte Carlo) with caching
# -----------------------------
@lru_cache(maxsize=4096)  # Increased cache size
def cached_equity_query(hole: Tuple[str, ...], community: Tuple[str, ...], num_opponents: int, stage: str, sims: int=1000) -> float:
    """
    Cache key expanded to include stage for better hits.
    """
    return simulate_equity_core(list(hole), list(community), num_opponents, stage, sims)

def simulate_equity(hero_cards: List[str], community_cards: List[str], num_opponents: int=1, stage: str="flop", sims: int=1000, deterministic: bool=False) -> float:
    """
    Public function. Uses cached queries when possible.
    hero_cards, community_cards: lists of strings
    deterministic: if True, iterate through combinations (slow for many unknowns) but deterministic.
    """
    # Normalizing tuple keys for cache
    key = (tuple(sorted(hero_cards)), tuple(sorted(community_cards)), num_opponents, stage, sims)
    try:
        return cached_equity_query(key[0], key[1], num_opponents, stage, sims)
    except Exception:
        # fallback
        return simulate_equity_core(hero_cards, community_cards, num_opponents, stage, sims, deterministic)

def single_sim(hero_cards, community_cards, num_opponents, deck, remaining_board_slots):
    draw = random.sample(deck, remaining_board_slots + 2 * num_opponents)
    opp_hands = [draw[i*2:(i+1)*2] for i in range(num_opponents)]
    board = community_cards + draw[2*num_opponents:]
    hero_score = best_five_from_seven(hero_cards + board)
    opp_scores = [best_five_from_seven(h + board) for h in opp_hands]
    best_opp = max(opp_scores, key=lambda s: (s[0], s[1]))
    cmp = compare_hands(hero_score, best_opp)
    return 1 if cmp > 0 else 0.5 if cmp == 0 else 0

def simulate_equity_core(hero_cards: List[str], community_cards: List[str], num_opponents: int=1, stage: str="flop", sims: int=1000, deterministic: bool=False) -> float:
    """
    Monte Carlo simulation: return win probability (ties counted as half).
    Optimizations:
     - early outs for made hands vs board (e.g., already best possible)
     - use random sampling unless deterministic requested
    """
    known = set(hero_cards + community_cards)
    deck = [c for c in DECK if c not in known]
    remaining_board_slots = 5 - len(community_cards)
    hero_cards_local = list(hero_cards)

    # Adjust sims dynamically based on stage
    if stage == "flop":
        sims = 2000
    elif stage == "turn":
        sims = 1500
    else:  # river
        sims = 1000
    sims = min(sims, 5000 // (num_opponents + 1))  # Scale down for multiway

    # trivial cases
    if remaining_board_slots == 0 and num_opponents == 0:
        # showdown with no opponents
        return 1.0

    wins_plus_half_ties = 0
    total = sims

    # if deterministic and combinatorially small, iterate combos
    max_comb = math.comb(len(deck), remaining_board_slots + 2 * num_opponents)
    if deterministic and max_comb <= sims:
        # iterate through all combos
        for draw in itertools.combinations(deck, remaining_board_slots + 2 * num_opponents):
            draw = list(draw)
            opp_hands = [draw[i*2:(i+1)*2] for i in range(num_opponents)]
            board = community_cards + draw[2*num_opponents:]
            hero_score = best_five_from_seven(hero_cards_local + board)
            opp_scores = [best_five_from_seven(h + board) for h in opp_hands]
            best_opp = max(opp_scores, key=lambda s: (s[0], s[1]))
            cmp = compare_hands(hero_score, best_opp)
            if cmp > 0:
                wins_plus_half_ties += 1
            elif cmp == 0:
                wins_plus_half_ties += 0.5
        total = max(1, max_comb)
        return wins_plus_half_ties / total

    # Monte Carlo random sampling with parallelization
    args = [(hero_cards_local, community_cards, num_opponents, deck, remaining_board_slots) for _ in range(sims)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda a: single_sim(*a), args))
    wins_plus_half_ties = sum(results)

    return wins_plus_half_ties / total

# -----------------------------
# Opponent Model
# -----------------------------
class OpponentModel:
    def __init__(self, history_len: int = 200, decay_factor: float = 0.95):
        self.stats = defaultdict(lambda: {
            "hands_seen": 0,
            "vpip": 0,  # voluntary put money in pot
            "pfr": 0,   # preflop raise
            "bets": 0,
            "raises": 0,
            "calls": 0,
            "folds": 0,
            "cbets": 0,
            "fold_to_cbet": 0,
            "showdowns": 0,
            "three_bet": 0,
            "wtsd": 0,
            "pos_aggression": defaultdict(float),
            "recent_actions": deque(maxlen=history_len),
            "stack_samples": deque(maxlen=history_len),
            "positions": defaultdict(int),
        })
        self.decay_factor = decay_factor

    def observe_action(self, pid: str, stage: str, action: str, position: str = "mp", stack: float = None, previous_action: str = None):
        s = self.stats[pid]
        # Apply decay to existing stats before update
        for key in ["vpip", "pfr", "bets", "raises", "calls", "folds", "cbets", "fold_to_cbet", "showdowns", "three_bet", "wtsd"]:
            s[key] *= self.decay_factor

        s["recent_actions"].append((stage, action, position))
        if stack is not None:
            s["stack_samples"].append(stack)
        s["positions"][position] += 1
        # update granular counters
        if stage == "preflop":
            if action in ("call", "raise", "bet"):
                s["vpip"] += 1
                s["hands_seen"] += 1
            if action == "raise":
                s["pfr"] += 1
                if previous_action == "raise":
                    s["three_bet"] += 1
        if action == "bet":
            s["bets"] += 1
        elif action == "raise":
            s["raises"] += 1
        elif action == "call":
            s["calls"] += 1
        elif action == "fold":
            s["folds"] += 1

        # Update position aggression
        aggression_score = 1 if action in ("bet", "raise") else 0 if action == "call" else -1 if action == "fold" else 0
        s["pos_aggression"][position] = (s["pos_aggression"][position] * s["positions"][position] + aggression_score) / (s["positions"][position] + 1)

    def record_cbet(self, pid: str, did_cbet: bool):
        if did_cbet:
            self.stats[pid]["cbets"] += 1

    def record_fold_to_cbet(self, pid: str, folded: bool):
        if folded:
            self.stats[pid]["fold_to_cbet"] += 1

    def record_showdown(self, pid: str, won: bool = False):
        self.stats[pid]["showdowns"] += 1
        if won:
            self.stats[pid]["wtsd"] += 1

    def get_profile(self, pid: str) -> Dict[str, float]:
        s = self.stats[pid]
        hands = max(1, s["hands_seen"])
        vpip = s["vpip"] / hands
        pfr = s["pfr"] / hands
        aggression = (s["bets"] + s["raises"]) / max(1, s["calls"])
        cbets = s["cbets"] / hands
        fold_to_cbet = s["fold_to_cbet"] / max(1, s["cbets"])
        showdown = s["showdowns"] / hands
        three_bet = s["three_bet"] / hands
        wtsd = s["wtsd"] / max(1, s["showdowns"])
        avg_stack = (sum(s["stack_samples"]) / len(s["stack_samples"])) if s["stack_samples"] else 0
        pos_dist = {k: v / sum(s["positions"].values()) if s["positions"] else 0 for k, v in s["positions"].items()}
        return {
            "VPIP": vpip,
            "PFR": pfr,
            "Aggression": aggression,
            "CBet%": cbets,
            "FoldToCbet%": fold_to_cbet,
            "Showdown%": showdown,
            "ThreeBet%": three_bet,
            "WTSD%": wtsd,
            "AvgStack": avg_stack,
            "PosDist": pos_dist,
            "HandsSeen": hands
        }

    def fold_probability(self, pid: str, pot_odds: float = 1.0, stage: str = "flop") -> float:
        """
        Estimate fold probability for an opponent given pot odds and stage.
        Use opponent aggression and fold tendencies with a logistic mapping.
        pot_odds: to_call / (pot + to_call) (smaller means cheaper to call)
        Returns float 0..1.
        """
        prof = self.get_profile(pid)
        # Base tendencies
        aggression = min(3.0, prof["Aggression"])  # cap
        fold_to_cbet = prof["FoldToCbet%"]
        vpip = prof["VPIP"]
        # Compose a score: higher aggression => lower fold, higher fold_to_cbet => higher fold
        score = 0.3 + 0.5 * (1 - aggression / 3.0) + 0.4 * fold_to_cbet
        # consider pot odds: if pot odds small (cheap), fold prob decreases
        # map pot_odds [0..1] to multiplier
        po_factor = max(0.1, min(1.5, 1.0 + (0.5 - pot_odds)))
        raw = score * po_factor
        # clamp [0.05, 0.95]
        return max(0.05, min(0.95, raw))

# -----------------------------
# Preflop heuristic (Chen formula-like quick score)
# -----------------------------
def chen_score(hole_cards: List[str]) -> float:
    """
    Fast hand strength heuristic (Chen-style):
    - base score from high card, pair bonus, suited bonus, connectivity bonus
    Output roughly correlates with preflop strength. Higher is better.
    """
    assert len(hole_cards) == 2
    r1 = card_to_rank(hole_cards[0])
    r2 = card_to_rank(hole_cards[1])
    high = max(r1, r2)
    low = min(r1, r2)
    # convert rank index to Chen's base values: A=10, K=8, Q=7, J=6, T=5, 9=4.5, etc
    rank_base = [0]*13
    rank_base[RANK_TO_INT['A']] = 10
    rank_base[RANK_TO_INT['K']] = 8
    rank_base[RANK_TO_INT['Q']] = 7
    rank_base[RANK_TO_INT['J']] = 6
    rank_base[RANK_TO_INT['T']] = 5
    for i, r in enumerate("98765432"):
        rank_base[RANK_TO_INT[r]] = max(1, 4.5 - (i * 0.5))

    score = max(rank_base[high], 1)
    # pair bonus
    if r1 == r2:
        score = max(score * 2, 5)
    # suited bonus
    if card_to_suit(hole_cards[0]) == card_to_suit(hole_cards[1]):
        score += 2
    # connectivity bonus (closer ranks)
    gap = abs(r1 - r2)
    if gap == 0:
        pass
    elif gap == 1:
        score += 1
    elif gap == 2:
        score += 0.5
    elif gap == 3:
        score += 0.2
    else:
        score -= 0.5
    return score

# -----------------------------
# Board Texture Analysis
# -----------------------------
def analyze_board(community: List[str]) -> str:
    """
    Analyze board texture: 'dry' (few draws), 'wet' (many draws/flush/straight possible).
    """
    suits_count = defaultdict(int)
    for c in community:
        suits_count[card_to_suit(c)] += 1
    ranks = sorted([card_to_rank(c) for c in community])
    flush_draw = any(count >= 3 for count in suits_count.values())  # 3+ same suit
    straight_draw = False
    for i in range(len(ranks) - 1):
        if ranks[i+1] - ranks[i] <= 2:  # Close ranks for draws
            straight_draw = True
            break
    if flush_draw or straight_draw:
        return "wet"
    return "dry"

# -----------------------------
# Personalities
# -----------------------------
PERSONALITIES = {
    "aggressive": {
        "risk_aversion": 0.2,
        "bluff_multiplier": 1.5,
        "value_bet_multiplier": 1.2,
        "description": "Plays aggressively, bluffs often, raises big."
    },
    "passive": {
        "risk_aversion": 0.8,
        "bluff_multiplier": 0.5,
        "value_bet_multiplier": 0.8,
        "description": "Plays cautiously, calls more, bluffs rarely."
    },
    "balanced": {
        "risk_aversion": 0.5,
        "bluff_multiplier": 1.0,
        "value_bet_multiplier": 1.0,
        "description": "Balanced play, mixes bluffs and value bets appropriately."
    },
    "loose": {
        "risk_aversion": 0.3,
        "bluff_multiplier": 1.2,
        "value_bet_multiplier": 1.1,
        "description": "Plays many hands, aggressive in pots."
    },
    "tight": {
        "risk_aversion": 0.7,
        "bluff_multiplier": 0.7,
        "value_bet_multiplier": 0.9,
        "description": "Plays few hands, but aggressively when in."
    },
}

# -----------------------------
# Decision Engine (PokerAI)
# -----------------------------
class PokerAI:
    def __init__(self, opponent_model: OpponentModel, my_pid: str = "hero", personality: str = "balanced"):
        self.om = opponent_model
        self.pid = my_pid
        if personality not in PERSONALITIES:
            raise ValueError(f"Unknown personality: {personality}. Available: {list(PERSONALITIES.keys())}")
        pers = PERSONALITIES[personality]
        self.risk_aversion = pers["risk_aversion"]
        self.bluff_multiplier = pers["bluff_multiplier"]
        self.value_bet_multiplier = pers["value_bet_multiplier"]
        self.personality = personality

    def decide(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        game_state expected keys:
         - hero: pid (should match self.pid)
         - hole_cards: list[str] (2)
         - community_cards: list[str] (0..5)
         - pot: float
         - to_call: float
         - stage: str in {"preflop","flop","turn","river"}
         - players: list of pids (including hero)
         - stacks: dict pid->stacksize (effective stacks)
         - min_raise: float (optional)
         - position: str (optional)
        Returns dict with keys: action (fold/call/raise/check/bet), size, equity, ev_call, rationale
        """
        hero = game_state["hero"]
        hole = game_state["hole_cards"]
        community = game_state.get("community_cards", [])
        pot = float(game_state.get("pot", 0.0))
        to_call = float(game_state.get("to_call", 0.0))
        stage = game_state.get("stage", "preflop")
        players = list(game_state.get("players", []))
        stacks = game_state.get("stacks", {})
        min_raise = game_state.get("min_raise", pot * 0.5 if pot > 0 else 1.0)
        position = game_state.get("position", "mp")
        opponents = [p for p in players if p != hero]

        # Effective stack (min over hero and opponents)
        eff_stack = min([stacks.get(hero, float('inf'))] + [stacks.get(p, float('inf')) for p in opponents])

        # Dynamic risk aversion based on stack (adjusted by personality)
        base_risk = self.risk_aversion
        if eff_stack < 20:
            self.risk_aversion = max(0.2, base_risk - 0.3)  # More aggressive short stack
        elif eff_stack > 100:
            self.risk_aversion = min(0.8, base_risk + 0.2)  # Tighter deep stack

        # Preflop direct heuristic
        if stage == "preflop":
            score = chen_score(hole)
            # conservative thresholds modulated by stack depth and position
            early_position = position in ("utg", "ep")
            # adjust for stack depth (short stacks -> more pushy)
            stack_factor = 1.0
            if eff_stack < 20:  # 20 bb
                stack_factor = 1.2
            if score > 8 and not early_position:
                # strong hand: raise
                size = round(min(eff_stack, pot * 3 if pot > 0 else 2 * min_raise) * self.value_bet_multiplier, 2)
                rationale = f"Preflop strong Chen {score:.2f}, stack_factor {stack_factor:.2f}, personality {self.personality}"
                return {"action": "raise", "size": size, "equity": None, "ev_call": None, "rationale": rationale}
            elif score > 5:
                # medium: call or small raise depending on to_call and pot odds
                pot_odds = to_call / (pot + to_call) if to_call > 0 else 0
                if to_call == 0:
                    return {"action": "raise", "size": round(min(eff_stack, pot + min_raise) * self.value_bet_multiplier, 2),
                            "equity": None, "ev_call": None, "rationale": f"Preflop limp/raise with Chen {score:.2f}, personality {self.personality}"}
                # compute approximate equity vs one random (not exact)
                est_eq = 0.5 if score > 6 else 0.35
                ev_call = est_eq * (pot + to_call) - (1 - est_eq) * to_call
                if ev_call > 0:
                    return {"action": "call", "size": to_call, "equity": est_eq, "ev_call": ev_call,
                            "rationale": f"Preflop call: Chen {score:.2f}, EV {ev_call:.2f}, personality {self.personality}"}
                else:
                    return {"action": "fold", "size": 0, "equity": est_eq, "ev_call": ev_call,
                            "rationale": f"Preflop fold: Chen {score:.2f}, EV {ev_call:.2f}, personality {self.personality}"}
            else:
                # weak hand: fold unless very cheap to call / multiway limp value
                if to_call == 0:
                    return {"action": "call", "size": 0, "equity": None, "ev_call": None, "rationale": f"Preflop limp (free), personality {self.personality}"}
                pot_odds = to_call / (pot + to_call) if to_call > 0 else 0
                if pot_odds < 0.05:
                    return {"action": "call", "size": to_call, "equity": None, "ev_call": None, "rationale": f"Cheap preflop call, personality {self.personality}"}
                return {"action": "fold", "size": 0, "equity": None, "ev_call": None, "rationale": f"Preflop fold (weak hand), personality {self.personality}"}

        # Postflop (flop/turn/river)
        # 1) Estimate equity vs opponents (use 500-2000 sims depending on stage)
        sims = 1500 if stage == "flop" else 1000 if stage == "turn" else 800
        # limit sims if many opponents
        if len(opponents) > 2:
            sims = max(300, sims // (len(opponents)))
        eq = simulate_equity(hole, community, len(opponents), stage, sims=sims)
        # Multiway adjustment: scale equity down
        multiway_factor = 1.0 if len(opponents) <= 1 else max(0.4, 1.0 - 0.25 * (len(opponents)-1))
        eq *= multiway_factor
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0
        ev_call = eq * (pot + to_call) - (1 - eq) * to_call

        # fold equity: average fold prob
        fe = 0.0
        if opponents:
            fe = sum(self.om.fold_probability(o, pot_odds, stage) for o in opponents) / len(opponents)

        # Estimate hand strength category quickly (using evaluator)
        cat, tieb = best_five_from_seven(hole + community)
        hand_strength = cat  # 0..8 mapping

        # Board texture for bluff freq
        board_texture = analyze_board(community)

        # Basic rules
        if to_call == 0:
            # check or bet
            if hand_strength >= 4:  # straight or better
                # strong: bet for value
                size = round(min(eff_stack, pot * (0.6 + 0.4 * (hand_strength / 8.0)) * self.value_bet_multiplier), 2)
                rationale = f"Bet for value: strong made hand cat {hand_strength}, eq {eq:.2f}, personality {self.personality}"
                return {"action": "bet", "size": size, "equity": eq, "ev_call": None, "rationale": rationale}
            elif eq > 0.35 and random.random() < 0.4 * (1 - self.risk_aversion) * self.bluff_multiplier:
                # semi-bluff with decent equity
                size = round(min(eff_stack, pot * 0.4 * self.bluff_multiplier), 2)
                return {"action": "bet", "size": size, "equity": eq,
                        "ev_call": None, "rationale": f"Semi-bluff bet: eq {eq:.2f}, risk_aversion {self.risk_aversion}, personality {self.personality}"}
            else:
                return {"action": "check", "size": 0, "equity": eq, "ev_call": None, "rationale": f"Check (no to_call), personality {self.personality}"}

        # There is an amount to call
        # Strong value raise if we have two pair or better and positive EV
        if hand_strength >= 2 and ev_call > 0:
            # raise for value
            base_raise = pot * (0.6 + 0.3 * (hand_strength / 8.0)) * self.value_bet_multiplier
            size = round(min(eff_stack, base_raise), 2)
            rationale = f"Value raise: hand_strength {hand_strength}, eq {eq:.2f}, EV_call {ev_call:.2f}, personality {self.personality}"
            return {"action": "raise", "size": size, "equity": eq, "ev_call": ev_call, "rationale": rationale}

        # If calling is +EV, call
        if ev_call > 0 and eq >= 0.2:
            return {"action": "call", "size": to_call, "equity": eq, "ev_call": ev_call, "rationale": f"Call EV positive {ev_call:.2f}, personality {self.personality}"}

        # Consider bluff raise if fold equity * pot > to_call (i.e., expected fold profit)
        bluff_freq = (0.3 if board_texture == "dry" else 0.15) * self.bluff_multiplier
        bluff_expected = fe * pot - (1 - fe) * (pot * 0.5)  # approximate cost if called
        # scale by multiway_factor and risk aversion
        bluff_expected *= multiway_factor * (1 - self.risk_aversion)
        if bluff_expected > to_call and random.random() < max(0.2, bluff_freq):
            size = round(min(eff_stack, pot * (0.5 + fe) * self.bluff_multiplier), 2)
            return {"action": "raise", "size": size, "equity": eq, "ev_call": ev_call,
                    "rationale": f"Bluff raise: FE {fe:.2f}, bluff_expected {bluff_expected:.2f}, texture {board_texture}, personality {self.personality}"}

        # Default fold if negative EV and no good bluff/fold equity
        return {"action": "fold", "size": 0, "equity": eq, "ev_call": ev_call,
                "rationale": f"Fold: eq {eq:.2f}, EV {ev_call:.2f}, FE {fe:.2f}, personality {self.personality}"}

# -----------------------------
# Function to run the bot
# -----------------------------
def run_poker_bot(input_data: Any, personality: str = "balanced", mode: str = "bot", opponent_model: OpponentModel = None) -> Dict[str, Any]:
    """
    Importable function to run the poker bot.
    
    Args:
    - input_data: Game state as dict or JSON string.
    - personality: One of 'aggressive', 'passive', 'balanced', 'loose', 'tight'.
    - mode: 'bot' for autonomous decision, 'assistant' for suggestion (adds extra rationale).
    - opponent_model: Optional OpponentModel instance; creates new if None.
    
    Returns:
    - Decision dict (JSON-serializable).
    """
    if isinstance(input_data, str):
        game_state = json.loads(input_data)
    elif isinstance(input_data, dict):
        game_state = input_data
    else:
        raise ValueError("input_data must be dict or JSON string")

    if opponent_model is None:
        opponent_model = OpponentModel()

    ai = PokerAI(opponent_model, my_pid=game_state.get("hero", "hero"), personality=personality)
    decision = ai.decide(game_state)

    if mode == "assistant":
        # Add extra suggestion text
        decision["suggestion"] = f"As your assistant with {personality} personality: {decision['rationale']}. Recommended action: {decision['action']} {decision.get('size', '')}."
    
    return decision

# -----------------------------
# Demo and quick unit test
# -----------------------------
if __name__ == "__main__":
    random.seed(42)

    # quick evaluator test: Royal flush vs four of a kind
    rf = ["Ah", "Kh", "Qh", "Jh", "Th", "2d", "3c"]
    quad = ["Ad", "Ac", "As", "Ah", "Kd", "Qc", "Js"]
    print("Royal flush eval:", best_five_from_seven(rf))
    print("Quads eval:", best_five_from_seven(quad))
    print("Compare RF > Quads:", compare_hands(best_five_from_seven(rf), best_five_from_seven(quad)))

    # Example game state
    om = OpponentModel()
    om.observe_action("villainA", "preflop", "raise", position="ep", stack=150, previous_action="raise")
    om.observe_action("villainB", "preflop", "call", position="mp", stack=120)
    om.observe_action("villainA", "flop", "bet", position="ep")
    om.observe_action("villainB", "flop", "fold", position="mp")

    game_state = {
        "hero": "hero",
        "players": ["hero", "villainA", "villainB"],
        "hole_cards": ["Ah", "Kd"],
        "community_cards": ["7d", "8s", "2c"],
        "pot": 40.0,
        "to_call": 10.0,
        "stage": "flop",
        "stacks": {"hero": 200.0, "villainA": 150.0, "villainB": 120.0},
        "position": "co",
        "min_raise": 10.0
    }

    # Test run_poker_bot as bot
    rec_bot = run_poker_bot(game_state, personality="aggressive", mode="bot", opponent_model=om)
    print("\n🧠 Poker Bot Recommendation (Aggressive):")
    print(rec_bot)

    # Test run_poker_bot as assistant
    rec_assist = run_poker_bot(json.dumps(game_state), personality="passive", mode="assistant", opponent_model=om)
    print("\n🧠 Poker Assistant Recommendation (Passive):")
    print(rec_assist)

    for pid in ["villainA", "villainB"]:
        print(f"{pid} profile:", om.get_profile(pid))

    # Equity quick test
    eq = simulate_equity(["Ah", "Kd"], ["7d", "8s", "2c"], num_opponents=1, stage="flop", sims=800)
    print(f"\nEstimated equity (AhKd vs 1 random) on {game_state['community_cards']}: {eq:.3f}")

    # Additional tests
    print("\nBoard texture dry:", analyze_board(["7d", "8s", "2c"]))  # dry
    print("Board texture wet:", analyze_board(["7d", "8d", "9d"]))  # wet flush draw
