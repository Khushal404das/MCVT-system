<?php
declare(strict_types=1);

// Create or update an operator account:
//   php web/bin/create-user.php <username> [operator|admin]
// The password is asked for interactively (or read from MCVT_PASSWORD).
if (PHP_SAPI !== 'cli') {
    exit("CLI only\n");
}
require dirname(__DIR__) . '/bootstrap.php';

use Mcvt\App;

$username = $argv[1] ?? '';
$role = $argv[2] ?? 'operator';
if (!preg_match('/^[A-Za-z0-9_.@-]{3,64}$/', $username) || !in_array($role, ['operator', 'admin'], true)) {
    fwrite(STDERR, "Usage: php web/bin/create-user.php <username> [operator|admin]\n");
    exit(1);
}
$password = getenv('MCVT_PASSWORD') ?: '';
if ($password === '') {
    fwrite(STDOUT, "Password for $username: ");
    if (stripos(PHP_OS, 'WIN') !== 0) {
        system('stty -echo');
    }
    $password = trim((string) fgets(STDIN));
    if (stripos(PHP_OS, 'WIN') !== 0) {
        system('stty echo');
    }
    fwrite(STDOUT, "\n");
}
if (strlen($password) < 8) {
    fwrite(STDERR, "Password must be at least 8 characters.\n");
    exit(1);
}
$hash = password_hash($password, PASSWORD_DEFAULT);
App::db()->prepare(
    'INSERT INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)
     ON CONFLICT(username) DO UPDATE SET password_hash = excluded.password_hash, role = excluded.role'
)->execute([$username, $hash, $role, microtime(true)]);
echo "User '$username' saved with role '$role'.\n";
