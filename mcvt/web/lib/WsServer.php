<?php
declare(strict_types=1);

namespace Mcvt;

/**
 * WebSocket hub (RFC 6455) in plain PHP streams.
 *
 *  - The Python engine connects with role=engine and the shared ENGINE_TOKEN.
 *  - Browsers connect with a short-lived token issued to a signed-in user.
 *  - Engine -> browsers: JSON events + binary video frames (fan-out).
 *  - Browsers -> engine: "focus" commands (which vehicle each operator follows).
 *
 * Slow browsers never stall the engine: if a client's send buffer is full,
 * video frames for that client are dropped (events are always queued).
 */
final class WsServer
{
    private const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
    private const MAX_BUFFER = 3 * 1024 * 1024;
    private const MAX_MESSAGE = 16 * 1024 * 1024;

    /** @var resource */
    private $server;
    /** @var array<int,array<string,mixed>> */
    private array $clients = [];
    private ?int $engineId = null;
    /** @var array<string,string> last event of each cached type, replayed to new browsers */
    private array $cache = [];
    /** @var array<string,string> last frame per camera */
    private array $lastFrames = [];
    /** @var array<int,array<string,mixed>> */
    private array $recentLogs = [];

    public function __construct(private string $host, private int $port, private string $engineToken)
    {
    }

    public function run(): void
    {
        $addr = "tcp://{$this->host}:{$this->port}";
        $server = @stream_socket_server($addr, $errno, $errstr);
        if (!$server) {
            throw new \RuntimeException("Cannot listen on $addr: $errstr");
        }
        stream_set_blocking($server, false);
        $this->server = $server;
        $this->log("WebSocket hub listening on ws://{$this->host}:{$this->port}");

        while (true) {
            $read = [$this->server];
            $write = [];
            foreach ($this->clients as $c) {
                $read[] = $c['sock'];
                if ($c['out'] !== '') {
                    $write[] = $c['sock'];
                }
            }
            $except = null;
            if (@stream_select($read, $write, $except, 1) === false) {
                continue;
            }
            foreach ($read as $sock) {
                if ($sock === $this->server) {
                    $this->accept();
                } else {
                    $this->readFrom((int) $sock);
                }
            }
            foreach ($write as $sock) {
                $this->flush((int) $sock);
            }
            $this->reapStale();
        }
    }

    // ------------------------------------------------------------- connections
    private function accept(): void
    {
        $sock = @stream_socket_accept($this->server, 0, $peer);
        if (!$sock) {
            return;
        }
        stream_set_blocking($sock, false);
        stream_set_write_buffer($sock, 0);
        $this->clients[(int) $sock] = [
            'sock' => $sock, 'peer' => $peer, 'in' => '', 'out' => '', 'handshake' => false,
            'role' => null, 'user' => null, 'focus' => null, 'view' => null, 'grid' => [], 'frag' => '', 'fragOp' => 0,
            'since' => time(), 'outFrames' => 0,
        ];
    }

    private function close(int $id, string $why = ''): void
    {
        if (!isset($this->clients[$id])) {
            return;
        }
        $c = $this->clients[$id];
        @fclose($c['sock']);
        unset($this->clients[$id]);
        if ($id === $this->engineId) {
            $this->engineId = null;
            $this->log('Engine disconnected' . ($why ? " ($why)" : ''));
            $this->broadcast(json_encode(['type' => 'engine', 'status' => 'offline', 'ts' => microtime(true)]));
            unset($this->cache['engine']);
        } elseif ($c['role'] === 'browser') {
            $this->sendFocusSet();
        }
    }

    private function reapStale(): void
    {
        foreach ($this->clients as $id => $c) {
            if (!$c['handshake'] && time() - $c['since'] > 10) {
                $this->close($id, 'handshake timeout');
            }
        }
    }

    private function readFrom(int $id): void
    {
        $c = &$this->clients[$id];
        $data = @fread($c['sock'], 65536);
        if ($data === false || ($data === '' && feof($c['sock']))) {
            unset($c);
            $this->close($id, 'connection closed');
            return;
        }
        $c['in'] .= $data;
        if (strlen($c['in']) > self::MAX_MESSAGE + 1024) {
            unset($c);
            $this->close($id, 'message too large');
            return;
        }
        if (!$c['handshake']) {
            unset($c);
            $this->handshake($id);
            return;
        }
        unset($c);
        $this->decodeFrames($id);
    }

    private function handshake(int $id): void
    {
        $c = &$this->clients[$id];
        $end = strpos($c['in'], "\r\n\r\n");
        if ($end === false) {
            return;
        }
        $head = substr($c['in'], 0, $end);
        $c['in'] = substr($c['in'], $end + 4);
        $lines = explode("\r\n", $head);
        $request = array_shift($lines);
        $headers = [];
        foreach ($lines as $l) {
            if (str_contains($l, ':')) {
                [$k, $v] = explode(':', $l, 2);
                $headers[strtolower(trim($k))] = trim($v);
            }
        }
        if (!preg_match('#^GET\s+(\S+)\s+HTTP/1\.1#i', $request, $m) || empty($headers['sec-websocket-key'])) {
            $this->reject($id, 400, 'Bad Request');
            return;
        }
        parse_str((string) parse_url($m[1], PHP_URL_QUERY), $query);
        $token = (string) ($query['token'] ?? '');
        $role = ($query['role'] ?? '') === 'engine' ? 'engine' : 'browser';

        if ($role === 'engine') {
            if ($this->engineToken === '' || !hash_equals($this->engineToken, $token)) {
                $this->log("Rejected engine connection from {$c['peer']}: bad token");
                $this->reject($id, 403, 'Forbidden');
                return;
            }
        } else {
            $user = $this->userForToken($token);
            if ($user === null) {
                $this->reject($id, 403, 'Forbidden');
                return;
            }
            $c['user'] = $user;
        }
        $accept = base64_encode(sha1($headers['sec-websocket-key'] . self::GUID, true));
        $c['out'] .= "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            . "Sec-WebSocket-Accept: $accept\r\n\r\n";
        $c['handshake'] = true;
        $c['role'] = $role;
        unset($c);

        if ($role === 'engine') {
            if ($this->engineId !== null && $this->engineId !== $id) {
                $this->close($this->engineId, 'replaced by a new engine connection');
            }
            $this->engineId = $id;
            // A new engine session: drop state cached from the previous one.
            $this->cache = [];
            $this->lastFrames = [];
            $this->log('Engine connected');
            $this->sendFocusSet();
        } else {
            $this->log("Browser connected for user {$this->clients[$id]['user']}");
            $engineStatus = json_encode(['type' => 'engine', 'status' => $this->engineId !== null ? 'online' : 'offline']);
            $this->queue($id, $this->encode($engineStatus, 1));
            foreach ($this->cache as $msg) {
                $this->queue($id, $this->encode($msg, 1));
            }
            if ($this->recentLogs) {
                $this->queue($id, $this->encode(json_encode(['type' => 'log_history', 'items' => $this->recentLogs]), 1));
            }
            foreach ($this->lastFrames as $frame) {
                $this->queue($id, $this->encode($frame, 2));
            }
        }
        $this->decodeFrames($id);
    }

    private function reject(int $id, int $code, string $text): void
    {
        @fwrite($this->clients[$id]['sock'], "HTTP/1.1 $code $text\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
        $this->close($id, $text);
    }

    private function userForToken(string $token): ?string
    {
        if ($token === '' || !ctype_xdigit($token)) {
            return null;
        }
        try {
            $st = App::db()->prepare(
                'SELECT u.username FROM ws_tokens t JOIN users u ON u.id = t.user_id WHERE t.token = ? AND t.expires_at > ?'
            );
            $st->execute([$token, microtime(true)]);
            $name = $st->fetchColumn();
            return $name === false ? null : (string) $name;
        } catch (\Throwable $e) {
            $this->log('Token check failed: ' . $e->getMessage());
            App::resetDb();
            return null;
        }
    }

    // ------------------------------------------------------------------ frames
    private function decodeFrames(int $id): void
    {
        while (isset($this->clients[$id])) {
            $c = &$this->clients[$id];
            $buf = $c['in'];
            $n = strlen($buf);
            if ($n < 2) {
                return;
            }
            $b0 = ord($buf[0]);
            $b1 = ord($buf[1]);
            $fin = ($b0 & 0x80) !== 0;
            $op = $b0 & 0x0F;
            $masked = ($b1 & 0x80) !== 0;
            $len = $b1 & 0x7F;
            $pos = 2;
            if ($len === 126) {
                if ($n < 4) {
                    return;
                }
                $len = unpack('n', substr($buf, 2, 2))[1];
                $pos = 4;
            } elseif ($len === 127) {
                if ($n < 10) {
                    return;
                }
                $len = unpack('J', substr($buf, 2, 8))[1];
                $pos = 10;
            }
            if ($len > self::MAX_MESSAGE) {
                unset($c);
                $this->close($id, 'frame too large');
                return;
            }
            $mask = '';
            if ($masked) {
                if ($n < $pos + 4) {
                    return;
                }
                $mask = substr($buf, $pos, 4);
                $pos += 4;
            }
            if ($n < $pos + $len) {
                return;
            }
            $payload = substr($buf, $pos, $len);
            $c['in'] = substr($buf, $pos + $len);
            if ($masked && $len > 0) {
                $payload ^= str_repeat($mask, intdiv($len, 4) + 1);
                $payload = substr($payload, 0, $len);
            }
            if ($c['role'] === 'browser' && !$masked) {
                unset($c);
                $this->close($id, 'unmasked client frame');
                return;
            }
            if ($op === 0x0) {               // continuation
                $c['frag'] .= $payload;
                if (!$fin) {
                    continue;
                }
                $payload = $c['frag'];
                $op = $c['fragOp'];
                $c['frag'] = '';
            } elseif (!$fin && ($op === 0x1 || $op === 0x2)) {
                $c['frag'] = $payload;
                $c['fragOp'] = $op;
                continue;
            }
            unset($c);
            switch ($op) {
                case 0x1:
                case 0x2:
                    $this->onMessage($id, $payload, $op === 0x2);
                    break;
                case 0x8:
                    $this->queue($id, $this->encode('', 0x8));
                    $this->flush($id);
                    $this->close($id, 'closed by peer');
                    return;
                case 0x9:
                    $this->queue($id, $this->encode($payload, 0xA));
                    break;
            }
        }
    }

    private function encode(string $payload, int $opcode): string
    {
        $len = strlen($payload);
        $head = chr(0x80 | $opcode);
        if ($len < 126) {
            $head .= chr($len);
        } elseif ($len < 65536) {
            $head .= chr(126) . pack('n', $len);
        } else {
            $head .= chr(127) . pack('J', $len);
        }
        return $head . $payload;
    }

    private function queue(int $id, string $bytes, bool $droppable = false): void
    {
        if (!isset($this->clients[$id])) {
            return;
        }
        if ($droppable && strlen($this->clients[$id]['out']) > self::MAX_BUFFER) {
            return; // client is too slow - skip this video frame
        }
        $this->clients[$id]['out'] .= $bytes;
        $this->flush($id);
    }

    private function flush(int $id): void
    {
        if (!isset($this->clients[$id]) || $this->clients[$id]['out'] === '') {
            return;
        }
        $c = &$this->clients[$id];
        $written = @fwrite($c['sock'], $c['out'], 262144);
        if ($written === false) {
            unset($c);
            $this->close($id, 'write failed');
            return;
        }
        $c['out'] = (string) substr($c['out'], $written);
        if (strlen($c['out']) > self::MAX_BUFFER * 4) {
            unset($c);
            $this->close($id, 'client too slow');
        }
    }

    private function broadcast(string $payload, bool $binary = false): void
    {
        $frame = $this->encode($payload, $binary ? 0x2 : 0x1);
        foreach ($this->clients as $id => $c) {
            if ($c['role'] === 'browser' && $c['handshake']) {
                $this->queue($id, $frame, $binary);
            }
        }
    }

    // ---------------------------------------------------------------- routing
    private function onMessage(int $id, string $payload, bool $binary): void
    {
        $role = $this->clients[$id]['role'];
        if ($role === 'engine') {
            if ($binary) {
                if (strlen($payload) > 4) {
                    $hlen = unpack('N', substr($payload, 0, 4))[1];
                    $head = json_decode(substr($payload, 4, $hlen), true);
                    if (is_array($head) && isset($head['cam'])) {
                        $this->lastFrames[(string) $head['cam']] = $payload;
                    }
                }
                $this->broadcast($payload, true);
                return;
            }
            $msg = json_decode($payload, true);
            if (!is_array($msg)) {
                return;
            }
            $type = (string) ($msg['type'] ?? '');
            if (in_array($type, ['cameras', 'vehicles', 'stats', 'engine'], true)) {
                $this->cache[$type] = $payload;
            }
            if ($type === 'log') {
                $this->recentLogs[] = $msg;
                $this->recentLogs = array_slice($this->recentLogs, -60);
            }
            $this->broadcast($payload);
            return;
        }
        if ($binary) {
            return; // browsers only send JSON commands
        }
        $msg = json_decode($payload, true);
        if (!is_array($msg)) {
            return;
        }
        switch ($msg['type'] ?? '') {
            case 'focus':
                $gid = $msg['gid'] ?? null;
                $this->clients[$id]['focus'] = (is_string($gid) && preg_match('/^[A-Za-z0-9_-]{1,32}$/', $gid)) ? $gid : null;
                $this->sendFocusSet();
                break;
            case 'view':
                $cam = $msg['cam'] ?? null;
                $this->clients[$id]['view'] = (is_string($cam) && preg_match('/^[A-Za-z0-9_-]{1,32}$/', $cam)) ? $cam : null;
                $grid = [];
                foreach ((array) ($msg['grid'] ?? []) as $g) {
                    if (is_string($g) && preg_match('/^[A-Za-z0-9_-]{1,32}$/', $g)) {
                        $grid[] = $g;
                    }
                }
                $this->clients[$id]['grid'] = array_slice($grid, 0, 64);
                $this->sendFocusSet();
                break;
            case 'ping':
                $this->queue($id, $this->encode(json_encode(['type' => 'pong', 'ts' => microtime(true)]), 1));
                break;
        }
    }

    private function sendFocusSet(): void
    {
        if ($this->engineId === null) {
            return;
        }
        $gids = [];
        $views = [];
        $grid = [];
        foreach ($this->clients as $c) {
            if ($c['role'] === 'browser' && $c['focus']) {
                $gids[$c['focus']] = true;
            }
            if ($c['role'] === 'browser' && !empty($c['view'])) {
                $views[$c['view']] = true;
            }
            if ($c['role'] === 'browser' && !empty($c['grid'])) {
                foreach ($c['grid'] as $g) {
                    $grid[$g] = true;
                }
            }
        }
        $this->queue($this->engineId, $this->encode(json_encode(['type' => 'focus_set', 'gids' => array_keys($gids)]), 1));
        // Cameras shown in someone's main view are streamed at full frame rate.
        $this->queue($this->engineId, $this->encode(json_encode(['type' => 'view_set', 'cams' => array_keys($views), 'grid' => array_keys($grid)]), 1));
    }

    private function log(string $msg): void
    {
        fwrite(STDOUT, '[' . date('Y-m-d H:i:s') . "] $msg\n");
    }
}
