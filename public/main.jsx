// main.jsx - React client (updated with strict turn enforcement, hidden admin controls, check button)
const { useState, useEffect, useRef } = React;

function useSocket(url, onMessage) {
    const wsRef = useRef(null);
    useEffect(() => {
        const ws = new WebSocket(url);
        wsRef.current = ws;
        ws.onopen = () => console.log('ws open');
        ws.onmessage = (e) => {
            const obj = JSON.parse(e.data);
            onMessage && onMessage(obj);
        };
        ws.onclose = () => console.log('ws close');
        return () => ws.close();
    }, [url]);
    const send = (obj) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(obj));
        }
    };
    return { send };
}

function App(){
    const [name, setName] = useState('');
    const [joined, setJoined] = useState(false);
    const [players, setPlayers] = useState([]);
    const [community, setCommunity] = useState([]);
    const [pot, setPot] = useState(0);
    const [phase, setPhase] = useState('lobby');
    const [dealerIndex, setDealerIndex] = useState(null);
    const [turnIndex, setTurnIndex] = useState(null);
    const [myId, setMyId] = useState(null);
    const [meHole, setMeHole] = useState([]);
    const [adminMode, setAdminMode] = useState(false);
    const [adminPass, setAdminPass] = useState('');
    const [chat, setChat] = useState([]);
    const [roundId, setRoundId] = useState(0);
    const [currentBet, setCurrentBet] = useState(0);
    const [SB, setSB] = useState(10);
    const [BB, setBB] = useState(20);
    const [bettingStarted, setBettingStarted] = useState(false);

    const { send } = useSocket((location.origin.replace(/^http/, 'ws')) + '/ws', (msg) => {
        if (msg.type === 'joined') {
            setMyId(msg.player.id);
        } else if (msg.type === 'state') {
            setPlayers(msg.players || []);
            setPot(msg.pot || 0);
            setCommunity(msg.community || []);
            setPhase(msg.phase || 'lobby');
            setDealerIndex(msg.dealerIndex);
            setTurnIndex(msg.turnIndex);
            setRoundId(msg.roundId || 0);
            setCurrentBet(msg.currentBet || 0);
            setSB(msg.SB || SB);
            setBB(msg.BB || BB);
            setBettingStarted(!!msg.bettingRoundStarted);
        } else if (msg.type === 'your_hole') {
            setMeHole(msg.hole || []);
        } else if (msg.type === 'admin_auth_result') {
            if (msg.ok) {
                setAdminMode(true);
                appendChat('admin access granted');
            } else {
                appendChat('admin access denied');
            }
        } else if (msg.type === 'kicked') {
            appendChat('You have been kicked by admin');
            setTimeout(()=>window.location.reload(), 1500);
        } else if (msg.type === 'error') {
            appendChat('Error: ' + (msg.message || 'unknown'));
        } else if (msg.type === 'showdown_result') {
            appendChat(`Showdown winners: ${msg.winners.map(w=>w.name).join(', ')} (share: ${msg.potShare})`);
        } else if (msg.type === 'auto_fold_win') {
            appendChat(`${msg.winner.name} won ${msg.pot} when everyone else folded`);
        }
    });

    function appendChat(t) { setChat(c => [...c, t].slice(-80)); }

    function join() {
        if (!name) return;
        send({ type: 'join', name });
        setJoined(true);
        appendChat(`Joined as ${name}`);
    }

    function adminAuth() {
        send({ type: 'admin_auth', pass: adminPass });
    }

    function adminCmd(cmd, opts={}) {
        send({ type: 'admin_cmd', cmd, ...opts });
    }

    function action(act, amt) {
        send({ type: 'action', action: act, amount: amt });
    }

    const meIndex = players.findIndex(p => p.id === myId);
    const me = players.find(p => p.id === myId) || {};

    const isMyTurn = meIndex === turnIndex;
    const canAct = (phase !== 'lobby') && isMyTurn && !(me.folded) && (me.chips > 0 || (me.currentBet > 0));

    function doBet() {
        const val = prompt('Enter bet amount', String(Math.max(1, (currentBet || 0) + 20)));
        if (!val) return;
        const amt = parseInt(val, 10);
        if (isNaN(amt) || amt <= 0) return alert('Invalid amount');
        action('bet', amt);
    }

    function doAllIn() {
        action('allin');
    }

    function doCall() {
        action('call');
    }

    function doCheck() {
        action('check');
    }

    function doFold() {
        action('fold');
    }

    return (
        <div className="app">
            <div className="left">
                <div className="header">
                    <div>
                        <div className="title">Minimal Hold'em</div>
                        <div className="small">Black &amp; Gold prototype — real-time (WebSocket)</div>
                    </div>
                    <div className="badge">Phase: {phase}</div>
                </div>

                {!joined ? (
                    <div className="lobby">
                        <input className="input" placeholder="Enter display name" value={name} onChange={e=>setName(e.target.value)} />
                        <div className="controls">
                            <button className="btn gold" onClick={join}>Join Lobby</button>
                        </div>
                        <div className="footer">Players start with 1000 chips. Small/Big blinds: {SB}/{BB}. Admin starts rounds.</div>
                    </div>
                ) : (
                    <>
                        <div className="table-view">
                            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                                <div>Pot: <strong style={{color:'#fff'}}>{pot}</strong></div>
                                <div className="small">Round #{roundId}</div>
                            </div>

                            <div style={{display:'flex',justifyContent:'center'}}>
                                <div className="community">
                                    {community.map((c,i)=><div className="card" key={i}>{renderCard(c)}</div>)}
                                    {Array.from({length:5-community.length}).map((_,i)=><div className="card" key={'empty'+i}>?</div>)}
                                </div>
                            </div>

                            <div className="center-line"></div>

                            <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
                                {players.map((p, idx) => {
                                    // Determine role labels
                                    let role = '';
                                    if (idx === dealerIndex) role = 'Dealer';
                                    else if (idx === (dealerIndex + 1) % players.length) role = 'Small Blind';
                                    else if (idx === (dealerIndex + 2) % players.length) role = 'Big Blind';
                                    else if (players.length > 3) {
                                        const utgIndex = (dealerIndex + 3) % players.length;
                                        if (idx === utgIndex) role = 'UTG';
                                        else if (idx === (utgIndex + 1) % players.length) role = 'CO';
                                        else if (idx === (utgIndex + 2) % players.length) role = 'MP';
                                        else if (idx === (utgIndex + 3) % players.length) role = 'BTN';
                                    }

                                    return (
                                        <div
                                            key={p.id}
                                            className="player"
                                            style={{
                                                minWidth: 160,
                                                border: idx === turnIndex
                                                    ? `1px solid ${idx === meIndex ? 'var(--gold)' : 'rgba(255,255,255,0.08)'}`
                                                    : 'none'
                                            }}
                                        >
                                            <div>
                                                <div className="name">
                                                    {p.name}{' '}
                                                    <span className="role">{role ? `(${role})` : ''}</span>
                                                </div>
                                                <div className="chips">Chips: {p.chips}</div>
                                                <div className="small">Bet: {p.currentBet}</div>
                                            </div>
                                            <div style={{ textAlign: 'right' }}>
                                                <div className="small">{p.folded ? 'Folded' : (idx === turnIndex ? 'Turn' : '')}</div>
                                                <div className="small">Seat {p.seat}</div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>

                            <div className="me">
                                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                                    <div>
                                        <div style={{fontWeight:700,color:'#fff'}}>{me.name || 'You'}</div>
                                        <div className="small">Chips: {me.chips ?? 0}</div>
                                    </div>
                                    <div style={{display:'flex',gap:8}}>
                                        <div className="card">{renderCard(meHole[0])}</div>
                                        <div className="card">{renderCard(meHole[1])}</div>
                                    </div>
                                </div>

                                <div style={{marginTop:8,display:'flex',gap:8,alignItems:'center'}}>
                                    <button className="btn" onClick={doFold} disabled={!canAct}>Fold</button>
                                    <button className="btn" onClick={doCheck} disabled={!canAct || (me.currentBet < currentBet)}>Check</button>
                                    <button className="btn" onClick={doCall} disabled={!canAct || (me.currentBet >= currentBet)}>Call</button>
                                    <button className="btn" onClick={doBet} disabled={!canAct}>Bet / Raise</button>
                                    <button className="btn small" onClick={doAllIn} disabled={!canAct}>All-in</button>
                                </div>
                                <div style={{marginTop:8}}>
                                    <div className="small">Current to call: {currentBet}</div>
                                    <div className="small">Your bet: {me.currentBet}</div>
                                </div>
                            </div>

                        </div>
                    </>
                )}
            </div>

            <div className="right">
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                    <div style={{fontWeight:700,color: 'var(--gold)'}}>Controls</div>
                    <div className="small">Admin: {adminMode ? 'ON' : 'OFF'}</div>
                </div>

                <div style={{marginTop:12}}>
                    <div className="small">Players</div>
                    <div className="player-list" style={{marginTop:8}}>
                        {players.map(p => <div className="player" key={'pl'+p.id}><div className="name">{p.name}</div><div className="chips">{p.chips}</div></div>)}
                    </div>
                </div>

                <div className="center-line"></div>

                <div>
                    <div className="small">Admin Panel</div>
                    <input className="input" placeholder="admin password" type="password" value={adminPass} onChange={e=>setAdminPass(e.target.value)} />
                    <div style={{display:'flex',gap:8,marginTop:8}}>
                        <button className="btn" onClick={adminAuth}>Unlock</button>
                        {/* Admin controls only visible when authenticated */}
                        {adminMode && <>
                            <button className="btn gold" onClick={()=>adminCmd('start_round')}>Start Round</button>
                            <button className="btn" onClick={()=>adminCmd('advance')}>Advance Phase</button>
                            <button className="btn" onClick={()=>adminCmd('reset_all')}>Reset All</button>
                        </>}
                    </div>

                    {adminMode && (
                        <div style={{marginTop:8}}>
                            <div className="small">Admin Actions</div>
                            <div style={{display:'flex',gap:8,marginTop:6}}>
                                <select id="kickSelect" className="input" style={{width:'100%'}}>
                                    <option value="">Select player to kick</option>
                                    {players.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                                </select>
                                <button className="btn" onClick={()=>{
                                    const sel = document.getElementById('kickSelect').value;
                                    if(sel) adminCmd('kick', { playerId: sel });
                                }}>Kick</button>
                            </div>
                        </div>
                    )}
                </div>

                <div className="center-line"></div>

                <div>
                    <div className="small">Logs</div>
                    <div style={{height:220,overflow:'auto',background:'rgba(255,255,255,0.02)',padding:8,borderRadius:8,marginTop:8}}>
                        {chat.map((c,i)=><div key={i} style={{fontSize:13}}>{c}</div>)}
                    </div>
                </div>

            </div>
        </div>
    );
}

function renderCard(c) {
    if (!c) return '??';
    // card like "As" -> render with suit symbol
    const rank = c.slice(0, -1);
    const s = c.slice(-1);
    const suitMap = {s:'♠',h:'♥',d:'♦',c:'♣'};
    return `${rank}${suitMap[s] || s}`;
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
