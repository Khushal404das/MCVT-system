<?php
declare(strict_types=1);

namespace Mcvt;

use PDO;

/**
 * Shared application services: configuration, .env, database, sessions/auth.
 */
final class App
{
    private static ?array $config = null;
    private static ?PDO $pdo = null;
    public static string $root = '';

    public static function init(string $root): void
    {
        self::$root = rtrim(str_replace('\\', '/', $root), '/');
        self::loadEnv(self::$root . '/.env');
        date_default_timezone_set(@date_default_timezone_get() ?: 'UTC');
    }

    // ------------------------------------------------------------------ config
    public static function config(string $dotted = '', mixed $default = null): mixed
    {
        if (self::$config === null) {
            self::$config = Yaml::parseFile(self::$root . '/config.yaml');
        }
        if ($dotted === '') {
            return self::$config;
        }
        $node = self::$config;
        foreach (explode('.', $dotted) as $part) {
            if (!is_array($node) || !array_key_exists($part, $node)) {
                return $default;
            }
            $node = $node[$part];
        }
        return $node;
    }

    public static function path(string $dotted, string $default = ''): string
    {
        $p = (string) self::config($dotted, $default);
        $p = str_replace('\\', '/', $p);
        if (preg_match('#^([A-Za-z]:/|/)#', $p)) {
            return $p;
        }
        return self::$root . '/' . $p;
    }

    public static function env(string $key, ?string $default = null): ?string
    {
        $v = getenv($key);
        if ($v === false || $v === '') {
            return $_ENV[$key] ?? $default;
        }
        return $v;
    }

    private static function loadEnv(string $file): void
    {
        if (!is_file($file)) {
            return;
        }
        foreach (file($file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) as $line) {
            $line = trim($line);
            if ($line === '' || $line[0] === '#' || !str_contains($line, '=')) {
                continue;
            }
            [$k, $v] = array_map('trim', explode('=', $line, 2));
            $v = trim($v, "\"'");
            if (getenv($k) === false) {
                putenv("$k=$v");
                $_ENV[$k] = $v;
            }
        }
    }

    /** Create .env with a random engine token if needed (shared secret with the Python engine). */
    public static function ensureEngineToken(): string
    {
        $token = self::env('ENGINE_TOKEN');
        if ($token) {
            return $token;
        }
        $token = bin2hex(random_bytes(24));
        file_put_contents(self::$root . '/.env', "\nENGINE_TOKEN=$token\n", FILE_APPEND);
        putenv("ENGINE_TOKEN=$token");
        $_ENV['ENGINE_TOKEN'] = $token;
        return $token;
    }

    // ---------------------------------------------------------------- database
    public static function db(): PDO
    {
        if (self::$pdo === null) {
            $file = self::path('paths.database', 'storage/mcvt.sqlite');
            if (!is_dir(dirname($file))) {
                mkdir(dirname($file), 0770, true);
            }
            $pdo = new PDO('sqlite:' . $file, null, null, [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_TIMEOUT => 30,
            ]);
            $pdo->exec('PRAGMA journal_mode=WAL');
            $pdo->exec('PRAGMA busy_timeout=30000');
            $pdo->exec('PRAGMA foreign_keys=ON');
            $pdo->exec(file_get_contents(self::$root . '/schema.sql'));
            self::$pdo = $pdo;
        }
        return self::$pdo;
    }

    public static function resetDb(): void
    {
        self::$pdo = null;
    }

    // -------------------------------------------------------------------- auth
    public static function startSession(): void
    {
        if (session_status() === PHP_SESSION_ACTIVE) {
            return;
        }
        session_name('MCVTSESS');
        // Sessions live in the project's storage (paths.sessions), not in the PHP install's temp
        // folder: removing / reinstalling PHP (e.g. XAMPP) must not break sign-in.
        $dir = self::path('paths.sessions', 'storage/sessions');
        if (!is_dir($dir)) {
            @mkdir($dir, 0700, true);
        }
        if (is_dir($dir) && is_writable($dir)) {
            session_save_path($dir);
        }
        session_set_cookie_params([
            'httponly' => true,
            'samesite' => 'Strict',
            'secure' => !empty($_SERVER['HTTPS']),
        ]);
        session_start();
    }

    /** @return array<string,mixed>|null */
    public static function user(): ?array
    {
        self::startSession();
        if (empty($_SESSION['uid'])) {
            return null;
        }
        $st = self::db()->prepare('SELECT id, username, role FROM users WHERE id = ?');
        $st->execute([$_SESSION['uid']]);
        return $st->fetch() ?: null;
    }

    /** NFR-S-02: every page / API call needs an authenticated user. */
    public static function requireUser(bool $api = false): array
    {
        $u = self::user();
        if ($u === null) {
            if ($api) {
                self::json(['error' => 'Please sign in first.'], 401);
            }
            header('Location: login.php');
            exit;
        }
        return $u;
    }

    public static function login(string $username, string $password): bool
    {
        $st = self::db()->prepare('SELECT id, password_hash FROM users WHERE username = ?');
        $st->execute([$username]);
        $row = $st->fetch();
        if (!$row || !password_verify($password, $row['password_hash'])) {
            return false;
        }
        self::startSession();
        session_regenerate_id(true);
        $_SESSION['uid'] = (int) $row['id'];
        $_SESSION['csrf'] = bin2hex(random_bytes(16));
        return true;
    }

    public static function csrfToken(): string
    {
        self::startSession();
        if (empty($_SESSION['csrf'])) {
            $_SESSION['csrf'] = bin2hex(random_bytes(16));
        }
        return $_SESSION['csrf'];
    }

    public static function checkCsrf(): void
    {
        $sent = $_SERVER['HTTP_X_CSRF_TOKEN'] ?? ($_POST['csrf'] ?? '');
        if (!is_string($sent) || !hash_equals(self::csrfToken(), $sent)) {
            self::json(['error' => 'Your session expired. Reload the page and try again.'], 403);
        }
    }

    public static function issueWsToken(int $userId): string
    {
        $token = bin2hex(random_bytes(24));
        $ttl = (int) self::config('auth.ws_token_ttl', 43200);
        $db = self::db();
        $db->prepare('DELETE FROM ws_tokens WHERE expires_at < ?')->execute([microtime(true)]);
        $db->prepare('INSERT INTO ws_tokens(token, user_id, expires_at) VALUES(?,?,?)')
            ->execute([$token, $userId, microtime(true) + $ttl]);
        return $token;
    }

    // ---------------------------------------------------------------- helpers
    public static function json(mixed $data, int $status = 200): never
    {
        http_response_code($status);
        header('Content-Type: application/json; charset=utf-8');
        header('Cache-Control: no-store');
        echo json_encode($data, JSON_UNESCAPED_SLASHES | JSON_INVALID_UTF8_SUBSTITUTE);
        exit;
    }

    public static function e(?string $s): string
    {
        return htmlspecialchars((string) $s, ENT_QUOTES, 'UTF-8');
    }
}
