<?php
declare(strict_types=1);

// GET api/vehicles.php?q=&camera=&limit=  - vehicle history from the database
require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
$db = App::db();
$limit = max(1, min(500, (int) ($_GET['limit'] ?? 200)));
$where = ['v.merged_into IS NULL'];
$args = [];
if (App::config('detection.moving_only', false)) {
    $where[] = "v.motion = 'moving'"; // parked vehicles are not reported
}
$q = trim((string) ($_GET['q'] ?? ''));
if ($q !== '') {
    $where[] = '(v.gid LIKE ? OR v.vclass LIKE ? OR v.color LIKE ?)';
    array_push($args, "%$q%", "%$q%", "%$q%");
}
$cam = trim((string) ($_GET['camera'] ?? ''));
if ($cam !== '') {
    $where[] = 'EXISTS (SELECT 1 FROM sightings s WHERE s.gid = v.gid AND s.camera_id = ?)';
    $args[] = $cam;
}
$sql = 'SELECT v.gid, v.vclass, v.color, v.motion, v.first_seen, v.last_seen, v.first_camera, v.last_camera,
               v.thumb_path, v.n_embeddings,
               (SELECT GROUP_CONCAT(DISTINCT camera_id) FROM sightings s WHERE s.gid = v.gid) AS cameras,
               (SELECT COUNT(*) FROM sightings s WHERE s.gid = v.gid) AS clips
        FROM vehicles v WHERE ' . implode(' AND ', $where) . ' ORDER BY v.last_seen DESC LIMIT ' . $limit;
$st = $db->prepare($sql);
$st->execute($args);
$rows = $st->fetchAll();
foreach ($rows as &$r) {
    $r['cameras'] = $r['cameras'] ? explode(',', $r['cameras']) : [];
    $r['thumb'] = $r['thumb_path'] ? 'media.php?thumb=' . rawurlencode($r['gid']) : null;
    unset($r['thumb_path']);
}
App::json(['items' => $rows]);
