<?php
declare(strict_types=1);

// GET api/vehicle.php?gid=VH-001 - full details: attributes, camera timeline, switches, requests
require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
$gid = (string) ($_GET['gid'] ?? '');
if (!preg_match('/^[A-Za-z0-9_-]{1,32}$/', $gid)) {
    App::json(['error' => 'Invalid vehicle id.'], 400);
}
$db = App::db();
// follow merges (a vehicle that turned out to be another one)
for ($i = 0; $i < 10; $i++) {
    $st = $db->prepare('SELECT merged_into FROM vehicles WHERE gid = ?');
    $st->execute([$gid]);
    $m = $st->fetchColumn();
    if (!$m) {
        break;
    }
    $gid = (string) $m;
}
$st = $db->prepare('SELECT * FROM vehicles WHERE gid = ?');
$st->execute([$gid]);
$v = $st->fetch();
if (!$v) {
    App::json(['error' => "Vehicle $gid was not found."], 404);
}
$v['thumb'] = $v['thumb_path'] ? 'media.php?thumb=' . rawurlencode($gid) : null;
unset($v['thumb_path']);

$st = $db->prepare(
    'SELECT s.camera_id, c.name AS camera_name, s.start_ts, s.end_ts, s.frames, s.best_score, s.width, s.height,
            s.start_clock, s.end_clock
     FROM sightings s LEFT JOIN cameras c ON c.id = s.camera_id WHERE s.gid = ? ORDER BY s.start_ts'
);
$st->execute([$gid]);
$sightings = $st->fetchAll();

$st = $db->prepare('SELECT from_cam, to_cam, reason, ts FROM switch_events WHERE gid = ? ORDER BY ts DESC LIMIT 30');
$st->execute([$gid]);
$switches = $st->fetchAll();

$st = $db->prepare('SELECT id, recipient, status, message, output_path, created_at, updated_at FROM jobs WHERE gid = ? ORDER BY id DESC LIMIT 10');
$st->execute([$gid]);
$jobs = $st->fetchAll();
foreach ($jobs as &$j) {
    $j['video'] = $j['output_path'] ? 'media.php?journey=' . (int) $j['id'] : null;
    unset($j['output_path']);
}

$cams = [];
foreach ($sightings as $s) {
    $cams[$s['camera_id']] = ($cams[$s['camera_id']] ?? 0) + max(0, $s['end_ts'] - $s['start_ts']);
}
App::json([
    'vehicle' => $v,
    'sightings' => $sightings,
    'camera_time' => $cams,
    'switches' => $switches,
    'jobs' => $jobs,
]);
