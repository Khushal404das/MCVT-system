<?php
declare(strict_types=1);

// POST api/journey.php  {gid, email}  -> queue a "full journey video" request (FR-UI-04)
// GET  api/journey.php?id=N           -> request status
require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

$user = App::requireUser(true);
$db = App::db();

if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $st = $db->prepare('SELECT id, gid, recipient, status, message, output_path, created_at, updated_at FROM jobs WHERE id = ?');
    $st->execute([(int) ($_GET['id'] ?? 0)]);
    $job = $st->fetch();
    if (!$job) {
        App::json(['error' => 'Request not found.'], 404);
    }
    $job['video'] = $job['output_path'] ? 'media.php?journey=' . (int) $job['id'] : null;
    unset($job['output_path']);
    App::json($job);
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    App::json(['error' => 'Method not allowed.'], 405);
}
App::checkCsrf();
$in = json_decode((string) file_get_contents('php://input'), true) ?: $_POST;
$gid = (string) ($in['gid'] ?? '');
$email = trim((string) ($in['email'] ?? ''));
if (!preg_match('/^[A-Za-z0-9_-]{1,32}$/', $gid)) {
    App::json(['error' => 'Invalid vehicle id.'], 400);
}
if ($email !== '' && !filter_var($email, FILTER_VALIDATE_EMAIL)) {
    App::json(['error' => 'Please enter a valid email address.'], 400);
}
$st = $db->prepare('SELECT COUNT(*) FROM vehicles WHERE gid = ?');
$st->execute([$gid]);
if (!(int) $st->fetchColumn()) {
    App::json(['error' => "Vehicle $gid is not in the database yet."], 404);
}
$now = microtime(true);
$db->prepare('INSERT INTO jobs(gid, recipient, status, message, requested_by, created_at, updated_at) VALUES(?,?,?,?,?,?,?)')
    ->execute([$gid, $email ?: null, 'queued', 'Waiting for the engine', $user['id'], $now, $now]);
$id = (int) $db->lastInsertId();
$db->prepare('INSERT INTO logs(ts, level, source, message) VALUES(?,?,?,?)')
    ->execute([$now, 'info', 'web', "{$user['username']} requested the full journey video of $gid" . ($email ? " for $email" : '') . " (request #$id)."]);
App::json(['id' => $id, 'status' => 'queued']);
