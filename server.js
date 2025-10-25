// server.js
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const ADMIN_PASS = process.env.ADMIN_PASS || 'adminpass';
const PORT = process.env.PORT || 3000;

const SB = parseInt(process.env.SMALL_BLIND || '25', 10);
const BB = parseInt(process.env.BIG_BLIND || '50', 10);

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: '/ws' });

/** Game state **/
let players = []; // {id, name, chips, ws, folded, hole:[], seat, currentBet, active}
let pot = 0;
let sidePots = []; // not fully fleshed but stubbed
let deck = [];
let community = [];
let phase = 'lobby'; // lobby, preflop, flop, turn, river, showdown
let dealerIndex = 0;
let turnIndex = 0; // index in players array for whose turn it is
let currentBet = 0; // amount to call
let roundId = 0;
let bettingRoundStarted = false;
let lastAggressorIndex = null; // index of last player to raise / bet
let checkCounter = 0;      // increments on checks/calls; reset on bet/raise/allin
let requiredChecks = 0;    // number of active players required to "check" to close the round

/* Utilities */
function makeDeck() {
  const suits = ['s','h','d','c']; // use letters internally
  const ranks = ['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
  const d = [];
  for (const s of suits) for (const r of ranks) d.push(r + s);
  return d;
}
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
}
function broadcast(obj) {
  const msg = JSON.stringify(obj);
  for (const p of players) {
    if (p.ws && p.ws.readyState === WebSocket.OPEN) {
      p.ws.send(msg);
    }
  }
}
function sendTo(ws, obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}
function publicPlayerInfo() {
  // Determine SB and BB relative to dealer
  const sbIndex = nextActiveIndex(dealerIndex);
  const bbIndex = nextActiveIndex(sbIndex);

  return players.map((p, idx) => {
    let role = '';

    if (idx === dealerIndex) role = 'Dealer';
    else if (idx === sbIndex) role = 'Small Blind';
    else if (idx === bbIndex) role = 'Big Blind';
    else {
      // Figure out if they’re UTG or other position
      // UTG = next active after BB
      const utgIndex = nextActiveIndex(bbIndex);
      if (idx === utgIndex) role = 'UTG';
      else role = 'Player';
    }

    return {
      id: p.id,
      name: p.name,
      chips: p.chips,
      folded: p.folded,
      seat: p.seat,
      currentBet: p.currentBet || 0,
      active: p.active !== false,
      role
    };
  });
}
function resetPlayerRoundState() {
  players.forEach((p, idx) => {
    p.hole = [];
    p.folded = false;
    p.currentBet = 0;
    p.active = p.chips > 0;
    p.seat = idx + 1; // seats are 1-based
  });
}
function compactPlayers() {
  // remove disconnected players that have no ws
  players = players.filter(p => !(p.ws === null && p.chips === 0));
}
function broadcastState() {
  broadcast({
    type: 'state',
    players: publicPlayerInfo(),
    pot,
    community,
    phase,
    dealerIndex,
    turnIndex,
    currentBet,
    roundId,
    SB,
    BB,
    bettingRoundStarted
  });
}

/* Poker core helpers */
function dealHoleCards() {
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < players.length; j++) {
      const p = players[(dealerIndex + 1 + j) % players.length];
      if (p && p.active) p.hole.push(deck.pop());
    }
  }
}
function startNewRound() {
  if (players.length === 0) return;

  // rotate dealer forward one seat (dealer rotates each new round)
  dealerIndex = (dealerIndex + 1) % players.length;

  resetPlayerRoundState();
  deck = makeDeck();
  shuffle(deck);
  community = [];
  pot = 0;
  sidePots = [];
  currentBet = 0;
  phase = 'preflop';
  roundId++;
  bettingRoundStarted = false;
  lastAggressorIndex = null;
  checkCounter = 0;
  requiredChecks = 0;

  // ensure at least two active players with chips
  const activeCount = players.filter(p => p.chips > 0).length;
  if (activeCount < 2) {
    broadcast({ type: 'error', message: 'Need at least 2 players with chips to start round' });
    phase = 'lobby';
    broadcastState();
    return;
  }

  // deal hole cards
  dealHoleCards();

  // find small blind and big blind (next active players after dealer)
  const sbIndex = nextActiveIndex(dealerIndex);
  const bbIndex = nextActiveIndex(sbIndex);

  // reset bets
  players.forEach(p => p.currentBet = 0);

  // post blinds
  postBlind(sbIndex, SB);
  postBlind(bbIndex, BB);

  currentBet = Math.max(SB, BB, players[bbIndex].currentBet); // ensure currentBet reflects BB (or larger if BB was all-in)
  // action starts on player after big blind (UTG)
  turnIndex = nextActiveIndex(bbIndex);

  // Setup check-counter semantics for phase closing: only count active players (not folded, not all-in)
  requiredChecks = players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0)).length;
  checkCounter = 0;

  bettingRoundStarted = true;
  // consider BB as last aggressor for preflop to prevent immediate closure unless everyone else acts
  lastAggressorIndex = bbIndex;

  broadcastState();

  // send private hole cards
  players.forEach(p => {
    if (p.hole && p.hole.length === 2) {
      sendTo(p.ws, { type: 'your_hole', hole: p.hole, roundId });
    }
  });
}

function postBlind(idx, amt) {
  const p = players[idx];
  if (!p || !p.active) return;
  const pay = Math.min(amt, p.chips);
  p.chips -= pay;
  p.currentBet = (p.currentBet || 0) + pay;
  pot += pay;
  if (p.chips === 0) p.active = false;
}

function nextActiveIndex(fromIdx) {
  if (players.length === 0) return 0;
  let i = (fromIdx + 1) % players.length;
  for (let tries=0; tries<players.length; tries++) {
    if (players[i] && players[i].chips > 0) return i;
    i = (i+1) % players.length;
  }
  // fallback
  return (fromIdx + 1) % players.length;
}

function allButOneFolded() {
  const active = players.filter(p => p.chips > 0 || p.currentBet > 0);
  const stillIn = active.filter(p => !p.folded);
  return stillIn.length <= 1;
}

function advancePhase() {
  if (phase === 'preflop') {
    deck.pop(); // burn
    community.push(deck.pop(), deck.pop(), deck.pop());
    phase = 'flop';
  } else if (phase === 'flop') {
    deck.pop();
    community.push(deck.pop());
    phase = 'turn';
  } else if (phase === 'turn') {
    deck.pop();
    community.push(deck.pop());
    phase = 'river';
  } else if (phase === 'river') {
    phase = 'showdown';
    performShowdown();
    return;
  }

  // reset per-phase betting trackers
  players.forEach(p => p.currentBet = 0);
  currentBet = 0;

  // reset check counting for the new betting round
  requiredChecks = players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0)).length;
  checkCounter = 0;
  bettingRoundStarted = true;

  // set turn to first active player after dealer (standard: first to act postflop is player left of dealer)
  turnIndex = nextActiveIndex(dealerIndex);
  broadcastState();
}

function isBettingRoundComplete() {
  const inPlayers = players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0));
  if (inPlayers.length <= 1) return true;

  // If everyone has matched currentBet or is all-in
  const allMatched = inPlayers.every(p => p.currentBet === currentBet || p.chips === 0);
  if (!allMatched) return false;

  // Only advance if turnIndex is back to last aggressor
  if (lastAggressorIndex === null) {
    // No one bet/raised: check if full loop done
    return true;
  } else {
    return turnIndex === nextActiveIndex(lastAggressorIndex);
  }
}

/* Hand evaluation: returns comparable score where higher is better.
   We'll implement a compact evaluator for: straight flush, quads, full house, flush, straight, trips, two pair, pair, high card.
   Score format: [rank(8..0), tiebreaker ranks...]
*/
function rankToValue(r) {
  if (r === 'A') return 14;
  if (r === 'K') return 13;
  if (r === 'Q') return 12;
  if (r === 'J') return 11;
  return parseInt(r, 10);
}
function parseCard(card) {
  // card like 'As' or '10h' etc.
  const suit = card.slice(-1);
  const rank = card.slice(0, -1);
  return { rank, suit, value: rankToValue(rank) };
}
function getAllCombos(arr, k) {
  const res = [];
  const n = arr.length;
  function backtrack(start, comb) {
    if (comb.length === k) {
      res.push(comb.slice());
      return;
    }
    for (let i = start; i < n; i++) {
      comb.push(arr[i]);
      backtrack(i+1, comb);
      comb.pop();
    }
  }
  backtrack(0, []);
  return res;
}
function evaluate5(cards) {
  // cards: array of 5 strings
  const parsed = cards.map(parseCard).sort((a,b)=>b.value - a.value);
  const values = parsed.map(p => p.value);
  const suits = parsed.map(p => p.suit);

  const counts = {};
  for (const v of values) counts[v] = (counts[v]||0) + 1;
  const countsArr = Object.entries(counts).map(([v,c])=>({v: parseInt(v,10), c})).sort((a,b) => b.c - a.c || b.v - a.v);

  const isFlush = suits.every(s => s === suits[0]);

  // Straight detection (consider wheel A-2-3-4-5)
  let uniqueVals = [...new Set(values)].sort((a,b)=>b-a);
  // handle wheel possibility by adding 1 for A as value 1
  let isStraight = false;
  let topStraightValue = 0;
  if (uniqueVals.length >= 5) {
    for (let i = 0; i <= uniqueVals.length - 5; i++) {
      const window = uniqueVals.slice(i, i+5);
      if (window[0] - window[4] === 4) {
        isStraight = true;
        topStraightValue = window[0];
        break;
      }
    }
  }
  // wheel: A(14),5,4,3,2
  if (!isStraight) {
    const hasA = uniqueVals.includes(14);
    const has5432 = [5,4,3,2].every(v=>uniqueVals.includes(v));
    if (hasA && has5432) { isStraight = true; topStraightValue = 5; }
  }

  // Straight flush
  if (isFlush && isStraight) {
    return {rank:8, tiebreak: [topStraightValue]};
  }

  // Four of a kind
  if (countsArr[0].c === 4) {
    const four = countsArr[0].v;
    const kicker = countsArr.find(x=>x.v !== four).v;
    return {rank:7, tiebreak: [four,kicker]};
  }

  // Full house
  if (countsArr[0].c === 3 && countsArr.length >1 && countsArr[1].c >=2) {
    const triple = countsArr[0].v;
    const pair = countsArr[1].v;
    return {rank:6, tiebreak:[triple,pair]};
  }

  // Flush
  if (isFlush) {
    return {rank:5, tiebreak: values};
  }

  // Straight
  if (isStraight) {
    return {rank:4, tiebreak:[topStraightValue]};
  }

  // Three of a kind
  if (countsArr[0].c === 3) {
    const triple = countsArr[0].v;
    const kickers = countsArr.slice(1).map(x=>x.v).slice(0,2);
    return {rank:3, tiebreak:[triple,...kickers]};
  }

  // Two pair
  if (countsArr[0].c === 2 && countsArr[1] && countsArr[1].c === 2) {
    const hi = countsArr[0].v;
    const lo = countsArr[1].v;
    const kicker = countsArr.slice(2).find(x=>x).v;
    return {rank:2, tiebreak:[hi,lo,kicker]};
  }

  // One pair
  if (countsArr[0].c === 2) {
    const pair = countsArr[0].v;
    const kickers = countsArr.slice(1).map(x=>x.v).slice(0,3);
    return {rank:1, tiebreak:[pair,...kickers]};
  }

  // High card
  return {rank:0, tiebreak: values.slice(0,5)};
}
function evaluateHand7(cards7) {
  // cards7: array of 7 card strings
  const combos = getAllCombos(cards7, 5);
  let best = null;
  for (const c of combos) {
    const eval5 = evaluate5(c);
    if (!best) { best = eval5; continue; }
    if (eval5.rank > best.rank) best = eval5;
    else if (eval5.rank === best.rank) {
      // compare tiebreakers lexicographically
      const a = eval5.tiebreak;
      const b = best.tiebreak;
      for (let i=0;i<Math.max(a.length,b.length);i++) {
        const av = a[i] || 0;
        const bv = b[i] || 0;
        if (av > bv) { best = eval5; break; }
        if (av < bv) break;
      }
    }
  }
  return best;
}

function performShowdown() {
  const results = players
    .filter(p => !p.folded && p.hole.length === 2)
    .map(p => ({ player: p, score: evaluateHand([...p.hole, ...community]) }));
  if (results.length === 0) {
    phase = 'lobby';
    dealerIndex = (dealerIndex + 1) % players.length;
    broadcastState();
    return;
  }
  let best = results[0];
  for (const r of results) {
    let winner = r;
    const a = r.score, b = best.score;
    for (let i=0; i<Math.max(a.length, b.length); i++) {
      const av = a[i] || 0;
      const bv = b[i] || 0;
      if (av > bv) { winner = r; break; }
      if (av < bv) { winner = best; break; }
    }
    if (winner === r) best = r;
  }
  const winners = results.filter(r => {
    if (r.score.rank !== best.score.rank) return false;
    const a = r.score.tiebreak, b = best.score.tiebreak;
    if (a.length !== b.length) return false;
    for (let i=0; i<a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  });
  const share = Math.floor(pot / winners.length);
  winners.forEach(w => w.player.chips += share);
  pot = 0;
  phase = 'lobby';
  dealerIndex = (dealerIndex + 1) % players.length;
  broadcast({
    type: 'showdown_result',
    winners: winners.map(w => ({ id: w.player.id, name: w.player.name })),
    community,
    potShare: share,
    allHands: players.map(p => ({
      id: p.id,
      name: p.name,
      hole: p.hole,
      folded: p.folded
    }))
  });
  broadcastState();
}

wss.on('connection', (ws) => {
  const id = Math.random().toString(36).slice(2,9);
  let attachedPlayer = null;

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === 'join') {
        const name = (msg.name || 'Player').slice(0,20);
        const seat = players.length + 1;
        const player = { id, name, chips: 1000, ws, folded: false, hole: [], seat, currentBet: 0, active: true };
        players.push(player);
        attachedPlayer = player;
        sendTo(ws, { type: 'joined', player: { id: player.id, name: player.name, chips: player.chips, seat } });
        broadcastState();
      }
      else if (msg.type === 'admin_auth') {
        const ok = msg.pass === ADMIN_PASS;
        sendTo(ws, { type: 'admin_auth_result', ok });
        if (ok) ws.isAdmin = true;
      }
      else if (msg.type === 'admin_cmd') {
        if (!ws.isAdmin) return sendTo(ws, { type: 'error', message: 'not admin' });
        const cmd = msg.cmd;
        if (cmd === 'start_round') {
          if (players.length < 2) return sendTo(ws, { type: 'error', message: 'Need at least 2 players' });
          startNewRound();
        } else if (cmd === 'advance') {
          advancePhase();
        } else if (cmd === 'reset_all') {
          players.forEach(p => { p.chips = 1000; p.hole = []; p.folded = false; p.currentBet = 0; p.active = true; });
          pot = 0; community = []; phase = 'lobby';
          broadcastState();
        } else if (cmd === 'kick') {
          const pid = msg.playerId;
          players = players.filter(p => {
            if (p.id === pid) {
              if (p.ws && p.ws.readyState === WebSocket.OPEN) sendTo(p.ws, { type: 'kicked' });
              return false;
            }
            return true;
          });
          broadcastState();
        } else if (cmd === 'adjust_chips') {
          const pid = msg.playerId;
          const amount = parseInt(msg.amount);
          if (!pid || isNaN(amount) || amount === 0) return sendTo(ws, { type: 'error', message: 'Invalid player or amount' });
          const player = players.find(p => p.id === pid);
          if (!player) return sendTo(ws, { type: 'error', message: 'Player not found' });
          const newChips = Math.max(0, player.chips + amount); // Prevent negative chips
          player.chips = newChips;
          player.active = newChips > 0;
          broadcast({ type: 'chat', message: `Admin adjusted ${player.name}'s chips by ${amount > 0 ? '+' : ''}${amount} to ${newChips}` });
          broadcastState();
        }
      }
      else if (msg.type === 'chat') { // New chat message handler
        const player = players.find(p => p.id === id);
        if (!player) return sendTo(ws, { type: 'error', message: 'Not joined' });
        const message = (msg.message || '').trim().slice(0, 200); // Limit length to prevent spam
        if (!message) return;
        broadcast({
          type: 'chat',
          from: player.name,
          message
        });
      }
      else if (msg.type === 'action') {
        // server enforces turns
        // actions: fold, check, call, bet(amount), allin
        const pIndex = players.findIndex(x => x.id === id);
        if (pIndex === -1) return;

        const actor = players[pIndex];
        if (phase === 'lobby') {
          return sendTo(ws, { type: 'error', message: 'No active hand' });
        }
        if (pIndex !== turnIndex) {
          return sendTo(ws, { type: 'error', message: 'Not your turn' });
        }
        if (actor.folded || actor.chips === 0) {
          // advance turn automatically
          turnIndex = nextActiveIndex(turnIndex);
          broadcastState();
          return;
        }

        const act = msg.action;

        // helper: compute number of currently "active-for-action" players
        function countActiveForAction() {
          return players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0)).length;
        }

        if (act === 'fold') {
          actor.folded = true;
          // folded players are not part of requiredChecks
        } else if (act === 'check') {
          if (actor.currentBet < currentBet) {
            return sendTo(ws, { type: 'error', message: 'Cannot check, must call or fold' });
          }
          // a check counts toward closing the betting round
          checkCounter += 1;
        } else if (act === 'call') {
          const toCall = currentBet - actor.currentBet;
          const actual = Math.min(toCall, actor.chips);
          actor.chips -= actual;
          actor.currentBet += actual;
          pot += actual;
          if (actor.chips === 0) actor.active = false;
          // calling counts toward the check-counter as a passive action
          checkCounter += 1;
        } else if (act === 'bet') {
          let amt = Math.max(0, Math.floor(msg.amount || 0));
          if (amt <= currentBet) {
            return sendTo(ws, { type: 'error', message: 'Bet must be greater than current bet (raise)' });
          }
          const toPut = amt - actor.currentBet;
          if (toPut > actor.chips) {
            return sendTo(ws, { type: 'error', message: 'Not enough chips' });
          }
          actor.chips -= toPut;
          actor.currentBet = amt;
          pot += toPut;
          currentBet = amt;

          // Raising/betting resets the check counter because players must respond to the raise
          checkCounter = 0;
          lastAggressorIndex = pIndex;
        } else if (act === 'allin') {
          const put = actor.chips;
          actor.currentBet += put;
          actor.chips = 0;
          pot += put;
          // If all-in increases the current bet, it's considered an aggressive action
          if (actor.currentBet > currentBet) {
            currentBet = actor.currentBet;
            lastAggressorIndex = pIndex;
            checkCounter = 0;
          } else {
            // If all-in merely calls, treat it as passive action
            checkCounter += 1;
          }
          actor.active = false;
        } else {
          return sendTo(ws, { type: 'error', message: 'unknown action' });
        }

        // advance turn to next active (non-folded) player
        let tries = 0;
        do {
          turnIndex = (turnIndex + 1) % players.length;
          tries++;
          // if we loop too much, break
          if (tries > players.length + 5) break;
        } while ((players[turnIndex].folded || players[turnIndex].chips === 0) && tries < players.length + 5);

        broadcastState();

        // check collapsing conditions: if all but one folded -> award pot
        if (allButOneFolded()) {
          // award pot to remaining player
          const winner = players.find(p => !p.folded);
          if (winner) {
            winner.chips += pot;
            broadcast({ type: 'auto_fold_win', winner: { id: winner.id, name: winner.name }, pot });
            pot = 0;
          }
          phase = 'lobby';
          dealerIndex = (dealerIndex + 1) % players.length;
          broadcastState();
          return;
        }

        // If only one player remains (everyone else folded), award pot immediately
        if (allButOneFolded()) {
          const winner = players.find(p => !p.folded);
          if (winner) {
            winner.chips += pot;
            broadcast({ type: 'auto_fold_win', winner: { id: winner.id, name: winner.name }, pot });
            pot = 0;
          }
          phase = 'lobby';
          dealerIndex = (dealerIndex + 1) % players.length;
          broadcastState();
          return;
        }

        // Recompute requiredChecks in case some players folded or went all-in
        requiredChecks = players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0)).length;

        // If checkCounter reaches requiredChecks, that means all active players had passive actions in a row — advance phase.
        if (requiredChecks > 0 && checkCounter >= requiredChecks) {
          // clear counter before advancing
          checkCounter = 0;
          advancePhase();
          return;
        }
      }
    } catch (e) {
      console.error('ws parse error', e);
      sendTo(ws, { type: 'error', message: 'bad message' });
    }
  });

  ws.on('close', () => {
    // detach player's ws but keep player in list (so their chips persist)
    const p = players.find(p => p.id === id);
    if (p) p.ws = null;
    players = players.filter(p => p.id !== id || p.ws !== null); // remove disconnected players
    broadcastState();
  });
});

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
