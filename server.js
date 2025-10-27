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

/* Poker hand evaluation utilities */
const rankValues = {
  '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
  '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
};

function parseCard(card) {
  const rank = card.slice(0, -1);
  const suit = card.slice(-1);
  return { rank: rankValues[rank], suit };
}

function getCombinations(arr, k, start = 0, curr = [], res = []) {
  if (curr.length === k) {
    res.push([...curr]);
    return res;
  }
  for (let i = start; i < arr.length; i++) {
    curr.push(arr[i]);
    getCombinations(arr, k, i + 1, curr, res);
    curr.pop();
  }
  return res;
}

function countRanks(ranks) {
  const c = {};
  ranks.forEach(r => c[r] = (c[r] || 0) + 1);
  return c;
}

function checkStraight(ranks) {
  const unique = [...new Set(ranks)].sort((a, b) => a - b);
  if (unique.length < 5) return false;
  if (unique[4] - unique[0] === 4) return true;
  // wheel straight
  if (unique.includes(14) && unique.includes(2) && unique.includes(3) && unique.includes(4) && unique.includes(5)) return true;
  return false;
}

function getHandValue(cards) {
  let ranks = cards.map(c => c.rank).sort((a, b) => b - a);
  const isFlush = cards.every(c => c.suit === cards[0].suit);
  const isStraight = checkStraight(ranks);
  const rankCounts = countRanks(ranks);
  // Sort ranks by count desc, then rank desc
  const countValues = Object.entries(rankCounts)
    .sort((a, b) => b[1] - a[1] || b[0] - a[0])
    .map(e => +e[0]);
  const counts = Object.values(rankCounts).sort((a, b) => b - a);

  let type = 0;
  let tiebreakers = ranks; // default for high card, flush

  if (counts[0] === 4) {
    type = 7; // quads
    tiebreakers = [countValues[0], countValues[1]];
  } else if (counts[0] === 3 && counts[1] === 2) {
    type = 6; // full house
    tiebreakers = [countValues[0], countValues[1]];
  } else if (counts[0] === 3) {
    type = 3; // trips
    tiebreakers = [countValues[0], countValues[1], countValues[2]];
  } else if (counts[0] === 2 && counts[1] === 2) {
    type = 2; // two pair
    tiebreakers = [countValues[0], countValues[1], countValues[2]];
  } else if (counts[0] === 2) {
    type = 1; // pair
    tiebreakers = [countValues[0], countValues[1], countValues[2], countValues[3]];
  } else {
    type = 0; // high card
  }

  if (isFlush) {
    type = isStraight ? 8 : 5; // straight flush or flush
  } else if (isStraight) {
    type = 4; // straight
  }

  if (isStraight || (isFlush && isStraight)) {
    let high = ranks[0];
    if (ranks[0] === 14 && ranks[4] === 2) high = 5;
    tiebreakers = [high];
  } else if (isFlush) {
    tiebreakers = ranks;
  }

  return { type, tiebreakers };
}

function compareHandValues(v1, v2) {
  if (v1.type > v2.type) return 1;
  if (v1.type < v2.type) return -1;
  const maxLen = Math.max(v1.tiebreakers.length, v2.tiebreakers.length);
  for (let i = 0; i < maxLen; i++) {
    const r1 = v1.tiebreakers[i] || 0;
    const r2 = v2.tiebreakers[i] || 0;
    if (r1 > r2) return 1;
    if (r1 < r2) return -1;
  }
  return 0;
}

function evaluatePlayerHand(p) {
  const seven = [...p.hole, ...community].map(parseCard);
  const combos = getCombinations(seven, 5);
  let best = { type: 0, tiebreakers: [] };
  for (const combo of combos) {
    const val = getHandValue(combo);
    if (compareHandValues(val, best) > 0) best = val;
  }
  return best;
}

/* Utilities */
function makeDeck() {
  const suits = ['s', 'h', 'd', 'c']; // use letters internally
  const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
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

/* Helper: determine "active" (eligible for dealing/acting) based on chips */
/* 
  isPlayerActiveForNewRound: used when starting a new round to decide who gets dealt cards / who is eligible for blinds.
  active-for-round checks used elsewhere (showdown, all-but-one-folded) must also include players who are all-in (currentBet > 0).
*/
function isPlayerActiveForNewRound(p) {
  return !!p && (p.chips > 0);
}

/* publicPlayerInfo: show roles relative to dealer.
   We'll use nextActiveIndex which by default will include currentBet players (all-ins) for role positioning.
*/
function publicPlayerInfo() {
  // Determine SB and BB relative to dealer using robust nextActiveIndex
  const sbIndex = nextActiveIndex(dealerIndex);
  const bbIndex = sbIndex !== -1 ? nextActiveIndex(sbIndex) : -1;

  return players.map((p, idx) => {
    let role = '';

    if (idx === dealerIndex) role = 'Dealer';
    else if (idx === sbIndex) role = 'Small Blind';
    else if (idx === bbIndex) role = 'Big Blind';
    else {
      // Figure out if they’re UTG or other position
      const utgIndex = bbIndex !== -1 ? nextActiveIndex(bbIndex) : -1;
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
      active: p.chips > 0,
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
  // remove disconnected players that have no ws and no chips
  players = players.filter(p => !(p.ws === null && p.chips === 0));
}

/* nextActiveIndex: find next player index after fromIdx that is "active"
   - by default (chipsOnly = false): return players that are either have chips > 0 OR have currentBet > 0 (i.e., in-pot / all-in participants are counted)
   - if chipsOnly = true: only return players with chips > 0 (used when deciding who can act or who should be dealt cards)
   returns -1 if none found.
*/
function nextActiveIndex(fromIdx, chipsOnly = false) {
  if (players.length === 0) return -1;
  let i = ((fromIdx + 1) % players.length + players.length) % players.length;
  for (let tries = 0; tries < players.length; tries++) {
    const p = players[i];
    if (!p) { i = (i + 1) % players.length; continue; }
    if (chipsOnly) {
      if (p.chips > 0) return i;
    } else {
      if (p.chips > 0 || (p.currentBet && p.currentBet > 0)) return i;
    }
    i = (i + 1) % players.length;
  }
  return -1;
}

function advanceIfUnable() {
  let loops = 0;
  while (loops < players.length * 2) { // safety limit
    if (players.length === 0) break;
    const actor = players[turnIndex];
    if (!actor) break;
    // actor must be connected, not folded, and have chips to be able to act
    if (actor.ws && actor.ws.readyState === WebSocket.OPEN && !actor.folded && actor.chips > 0) {
      break; // able to act
    } else {
      if (actor.ws === null || actor.ws.readyState !== WebSocket.OPEN) {
        // disconnected, auto-fold if not already folded
        if (!actor.folded) {
          actor.folded = true;
          broadcast({ type: 'chat', message: `${actor.name} disconnected, auto-folding` });
        }
      } else if (actor.chips === 0 && !actor.folded) {
        // all-in / out of chips, auto-pass (count as check/call)
        checkCounter++;
      }
      // advance turn to next player who actually has chips (only those with chips can be asked to act)
      const next = nextActiveIndex(turnIndex, true);
      if (next === -1) break;
      turnIndex = next;
    }
    loops++;
  }
}

function broadcastState() {
  if (phase !== 'lobby') {
    advanceIfUnable(); // auto-advance if current turn cannot act
  }
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
  // Deal only to players with chips > 0, in seat order starting after dealer.
  const activePlayersInOrder = [];
  for (let j = 0; j < players.length; j++) {
    const p = players[(dealerIndex + 1 + j) % players.length];
    if (isPlayerActiveForNewRound(p)) activePlayersInOrder.push(p);
  }
  // ensure hole arrays exist
  activePlayersInOrder.forEach(p => { if (!p.hole) p.hole = []; });

  for (let i = 0; i < 2; i++) {
    for (const p of activePlayersInOrder) {
      p.hole.push(deck.pop());
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

  // ensure at least two active players with chips (only players with chips are eligible for new rounds)
  const activeCount = players.filter(p => p.chips > 0).length;
  if (activeCount < 2) {
    broadcast({ type: 'error', message: 'Need at least 2 players with chips to start round' });
    phase = 'lobby';
    broadcastState();
    return;
  }

  // deal hole cards only to players with chips
  dealHoleCards();

  // find small blind and big blind (next active players after dealer); require chips > 0 for blinds
  const sbIndex = nextActiveIndex(dealerIndex, true);
  const bbIndex = (sbIndex !== -1) ? nextActiveIndex(sbIndex, true) : -1;

  // reset bets
  players.forEach(p => p.currentBet = 0);

  // post blinds (only if found valid indices)
  if (sbIndex !== -1) postBlind(sbIndex, SB);
  if (bbIndex !== -1) postBlind(bbIndex, BB);

  currentBet = Math.max(SB, BB, ...players.map(p => p.currentBet || 0)); // ensure currentBet reflects BB or larger
  // action starts on player after big blind (UTG). If bbIndex invalid, pick first active after dealer
  let firstToAct = (bbIndex !== -1) ? nextActiveIndex(bbIndex, true) : nextActiveIndex(dealerIndex, true);
  if (firstToAct === -1) {
    // Fallback: find any active player index (chips > 0)
    firstToAct = players.findIndex(p => p.chips > 0);
    if (firstToAct === -1) {
      // shouldn't happen because activeCount >= 2 checked earlier, but guard anyway
      phase = 'lobby';
      broadcast({ type: 'error', message: 'No active players to start round' });
      broadcastState();
      return;
    }
  }
  turnIndex = firstToAct;

  // Setup check-counter semantics for phase closing: only count active players (not folded, not all-in-only)
  requiredChecks = players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0)).length;
  checkCounter = 0;

  bettingRoundStarted = true;
  // consider BB as last aggressor for preflop to prevent immediate closure unless everyone else acts
  lastAggressorIndex = bbIndex;

  broadcastState();

  // send private hole cards
  players.forEach(p => {
    // only send to players who were dealt cards (chips > 0 and hole length === 2)
    if (p.hole && p.hole.length === 2 && p.ws) {
      sendTo(p.ws, { type: 'your_hole', hole: p.hole, roundId });
    }
  });
}

function postBlind(idx, amt) {
  const p = players[idx];
  if (!p || p.chips <= 0) return;
  const pay = Math.min(amt, p.chips);
  p.chips -= pay;
  p.currentBet = (p.currentBet || 0) + pay;
  pot += pay;
  if (p.chips === 0) p.active = false;
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
  const first = nextActiveIndex(dealerIndex, true); // choose players who actually have chips to act
  if (first !== -1) turnIndex = first;
  broadcastState();
}

function isBettingRoundComplete() {
  const inPlayers = players.filter(p => !p.folded && (p.chips > 0 || p.currentBet > 0)).length;
  return requiredChecks > 0 && checkCounter >= requiredChecks;
}

function performShowdown() {
  // reveal all hands for non-folded players who have hole cards (include all-in players)
  const participants = players.filter(p => !p.folded && (p.hole && p.hole.length >= 2));

  if (participants.length === 0) return;

  const allHands = participants.map(p => ({ id: p.id, name: p.name, hole: p.hole }));

  // evaluate hands
  participants.forEach(p => p.bestValue = evaluatePlayerHand(p));

  // find winners
  let maxValue = { type: 0, tiebreakers: [] };
  participants.forEach(p => {
    if (compareHandValues(p.bestValue, maxValue) > 0) {
      maxValue = p.bestValue;
    }
  });

  const winners = participants.filter(p => compareHandValues(p.bestValue, maxValue) === 0);

  const potShare = Math.floor(pot / winners.length);
  winners.forEach(w => w.chips += potShare);
  pot = 0;

  broadcast({
    type: 'showdown_result',
    winners: winners.map(w => ({ id: w.id, name: w.name })),
    potShare,
    allHands
  });

  // after short delay, start new round or back to lobby
  setTimeout(() => {
    startNewRound();
  }, 5000);
}

/* WebSocket handling */
let idCounter = 0;
wss.on('connection', (ws) => {
  let id = null;
  let isAdmin = false;

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data);

      if (msg.type === 'join') {
        const existing = players.find(p => p.name === msg.name && p.ws === null);
        if (existing) {
          existing.ws = ws;
          id = existing.id;
          sendTo(ws, { type: 'joined', player: { id, name: existing.name, chips: existing.chips } });
          broadcastState();
          return;
        }
        id = Date.now() + idCounter++;
        players.push({
          id,
          name: msg.name,
          chips: 1000,
          ws,
          folded: false,
          hole: [],
          seat: players.length + 1,
          currentBet: 0,
          active: true
        });
        sendTo(ws, { type: 'joined', player: { id, name: msg.name, chips: 1000 } });
        broadcastState();
      } else if (msg.type === 'admin_auth') {
        if (msg.pass === ADMIN_PASS) {
          isAdmin = true;
          sendTo(ws, { type: 'admin_auth_result', ok: true });
        } else {
          sendTo(ws, { type: 'admin_auth_result', ok: false });
        }
      } else if (msg.type === 'admin_cmd') {
        if (!isAdmin) return sendTo(ws, { type: 'error', message: 'Not admin' });
        const cmd = msg.cmd;
        if (cmd === 'start_round') {
          startNewRound();
        } else if (cmd === 'advance') {
          if (phase !== 'lobby' && phase !== 'showdown') advancePhase();
        } else if (cmd === 'reset_all') {
          players.forEach(p => p.chips = 1000);
          pot = 0;
          community = []; phase = 'lobby';
          broadcastState();
        } else if (cmd === 'kick') {
          // Ensure numeric id matching and proper removal
          const pid = Number(msg.playerId);
          if (isNaN(pid)) return sendTo(ws, { type: 'error', message: 'Invalid player id' });
          const toKick = players.find(p => p.id === pid);
          if (!toKick) return sendTo(ws, { type: 'error', message: 'Player not found' });
          // notify and close connection if open
          if (toKick.ws && toKick.ws.readyState === WebSocket.OPEN) {
            try { sendTo(toKick.ws, { type: 'kicked' }); } catch (e) {}
            try { toKick.ws.close(); } catch (e) {}
          }
          players = players.filter(p => p.id !== pid);
          broadcast({ type: 'chat', message: `Admin kicked ${toKick.name}` });
          broadcastState();
        } else if (cmd === 'adjust_chips') {
          const pid = Number(msg.playerId);
          const amount = parseInt(msg.amount, 10);
          if (isNaN(pid) || isNaN(amount)) return sendTo(ws, { type: 'error', message: 'Invalid player or amount' });
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
          // advance turn automatically to next player with chips > 0
          const next = nextActiveIndex(turnIndex, true);
          if (next !== -1) turnIndex = next;
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

        // advance turn to next active (non-folded) player who has chips (only players with chips can act)
        const next = nextActiveIndex(turnIndex, true);
        if (next !== -1) turnIndex = next;

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
    if (phase === 'lobby') compactPlayers();
    broadcastState(); // this will trigger advanceIfUnable if needed
  });
});

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});