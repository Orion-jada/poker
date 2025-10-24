# Online Texas Hold’em Poker

A real-time multiplayer Texas Hold’em poker game running on a local network. Built with **Node.js, Express, WebSocket, and React**, featuring an admin panel, rotating blinds, and realistic betting mechanics. Designed for **modern minimalist UI** (black and gold).

---

## **Features**

### 🎮 Gameplay

* Texas Hold’em poker rules.
* Real-time multiplayer on local network.
* Chips-based betting with blinds (Small Blind & Big Blind) and rotating dealer.
* Players can **fold, check, call, bet, or go all-in**.
* Betting rounds advance only when all active players have acted consecutively.
* Automatic pot awarding if all but one player folds.

### 🛠 Admin Panel

* Password-protected access.
* Start rounds, advance phases, reset chips, and kick players.
* Check button for quick phase progression.
* All administrative actions broadcast to all clients.

### 🖤 UI

* Modern minimalist design: **black and gold** theme.
* Displays player names, chips, hole cards (private), and community cards.
* Shows pot, blinds, current phase, and whose turn it is.

---

## **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/online-poker.git
cd online-poker
```

2. **Install dependencies**

```bash
npm install
```

3. **Start server**

```bash
node server.js
```

Optional environment variables:

* `PORT` – server port (default: `3000`)
* `ADMIN_PASS` – admin password (default: `adminpass`)
* `SMALL_BLIND` – small blind amount (default: `10`)
* `BIG_BLIND` – big blind amount (default: `20`)

4. **Open client**
   Open your browser to `http://localhost:3000`

---

## **Usage**

### **Player**

1. Enter a name and join the table.
2. Wait for the round to start.
3. Perform actions (fold, check, call, bet, all-in) when it’s your turn.
4. Watch the game progress with other players in real-time.

### **Admin**

1. Click “Admin Panel” and enter the admin password.
2. Use controls to:

    * Start a new round
    * Advance phases
    * Reset all chips
    * Kick a player
3. Optionally, use the “Check” button to proceed without manual phase advancement.

---

## **Game Mechanics**

* **Blinds:** Small blind posts first, big blind posts second. Blinds rotate each round.
* **Action Order:** Preflop starts after big blind. Post-flop starts from first active player left of dealer.
* **Betting Rounds:** Continue until all active players have matched the current bet or gone all-in.
* **Showdown:** Hands are evaluated, and the pot is distributed among winners. Supports split pots.

---

## **Development**

* Built with **Node.js + Express** for backend.
* **WebSockets** for real-time multiplayer.
* **React** for frontend UI.
* Minimalist styling in CSS (black and gold theme).
* Single-page application for simplicity: ideally **one JS, one CSS, one HTML** file.

---

## **License**

MIT License. Free to use and modify.

---

## **Future Improvements**

* Full side pot handling for multiple all-ins.
* Hand history and player statistics.
* Chat functionality.
* Support for larger tables and multiple game rooms.
