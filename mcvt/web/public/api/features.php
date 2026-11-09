<?php
declare(strict_types=1);

// GET api/features.php?gid=VH-001 - what the ReID model extracted for one vehicle (dashboard
// "Extracted features" panel):
//   attributes  colour + type stored with the id
//   embedding   the vehicle's mean ResNet50-IBN embedding (2048 numbers, L2-normalised)
//   samples[]   every stored embedding (camera, time, similarity to the mean)
//   pairwise    cosine similarity between the stored samples (how consistent the fingerprint is)
//   nearest[]   the most similar OTHER vehicles in the gallery (what an ID decision compares against)
require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
session_write_close();
$gid = (string) ($_GET['gid'] ?? '');
if (!preg_match('/^[A-Za-z0-9_-]{1,32}$/', $gid)) {
    App::json(['error' => 'Invalid vehicle id.'], 400);
}
$db = App::db();
for ($i = 0; $i < 10; $i++) { // follow merges
    $st = $db->prepare('SELECT merged_into FROM vehicles WHERE gid = ?');
    $st->execute([$gid]);
    $m = $st->fetchColumn();
    if (!$m) {
        break;
    }
    $gid = (string) $m;
}
$st = $db->prepare('SELECT gid, vclass, color, first_seen, last_seen, n_embeddings, thumb_path FROM vehicles WHERE gid = ?');
$st->execute([$gid]);
$v = $st->fetch();
if (!$v) {
    App::json(['error' => "Vehicle $gid was not found."], 404);
}

$maxSamples = (int) App::config('reid.gallery_size', 30);
$threshold = (float) App::config('reid.similarity_threshold', 0.75);
$handoff = (float) App::config('reid.handoff_threshold', $threshold);
// With the camera adapter on, stored embeddings are in the adapter's space and use its threshold.
$adapter = (string) App::config('reid.adapter', '');
if ($adapter !== '' && App::config('reid.adapter_threshold')) {
    $threshold = $handoff = (float) App::config('reid.adapter_threshold');
}

/** float32 blob -> unit vector (list of floats) */
$unpack = static function (string $blob): array {
    $x = array_values(unpack('g*', $blob));
    $n = 0.0;
    foreach ($x as $f) {
        $n += $f * $f;
    }
    $n = sqrt($n) ?: 1.0;
    foreach ($x as $k => $f) {
        $x[$k] = $f / $n;
    }
    return $x;
};
$dot = static function (array $a, array $b): float {
    $s = 0.0;
    foreach ($a as $k => $f) {
        $s += $f * $b[$k];
    }
    return $s;
};
$mean = static function (array $vecs) use ($dot): array {
    $m = array_fill(0, count($vecs[0]), 0.0);
    foreach ($vecs as $x) {
        foreach ($x as $k => $f) {
            $m[$k] += $f;
        }
    }
    $n = sqrt($dot($m, $m)) ?: 1.0;
    return array_map(static fn($f) => $f / $n, $m);
};

$st = $db->prepare('SELECT e.id, e.camera_id, e.ts, e.vec FROM embeddings e WHERE e.gid = ? ORDER BY e.id DESC LIMIT ' . $maxSamples);
$st->execute([$gid]);
$rows = array_reverse($st->fetchAll());
if ($rows) { // only samples of the current ReID setup (2048 plain / 256 with the camera adapter)
    $len = strlen((string) end($rows)['vec']);
    $rows = array_values(array_filter($rows, static fn($r) => strlen((string) $r['vec']) === $len));
}
$vecs = [];
$samples = [];
foreach ($rows as $r) {
    $vecs[] = $unpack((string) $r['vec']);
    $samples[] = ['id' => (int) $r['id'], 'camera' => $r['camera_id'], 'ts' => (float) $r['ts']];
}

$out = [
    'gid' => $gid,
    'attributes' => [
        'type' => $v['vclass'],
        'color' => $v['color'],
        'thumb' => $v['thumb_path'] ? 'media.php?thumb=' . rawurlencode($gid) : null,
    ],
    'model' => [
        'name' => 'ResNet50-IBN-a + GeM pooling + BNNeck (VeRi-776)'
            . ($vecs && count($vecs[0]) !== 2048 ? ' + camera adapter' : ''),
        'input' => App::config('reid.input_size', [256, 256]),
        'dim' => $vecs ? count($vecs[0]) : 0,
        'threshold' => $threshold,
        'handoff_threshold' => $handoff,
    ],
    'n_embeddings' => (int) $v['n_embeddings'],
    'samples' => [],
    'pairwise' => [],
    'embedding' => [],
    'nearest' => [],
];
if (!$vecs) {
    App::json($out);
}

$centre = $mean($vecs);
foreach ($samples as $k => $s) {
    $s['sim_to_mean'] = round($dot($vecs[$k], $centre), 3);
    $out['samples'][] = $s;
}
$n = count($vecs);
for ($a = 0; $a < $n; $a++) {
    $row = [];
    for ($b = 0; $b < $n; $b++) {
        $row[] = $a === $b ? 1.0 : round($dot($vecs[$a], $vecs[$b]), 3);
    }
    $out['pairwise'][] = $row;
}
$out['embedding'] = array_map(static fn($f) => round($f, 4), $centre);

// Nearest other vehicles: the gallery around this vehicle's time (same run of the engine).
$window = (float) App::config('reid.handoff_window_seconds', 90) + 3600;
$st = $db->prepare(
    'SELECT e.gid, e.vec FROM embeddings e JOIN vehicles v ON v.gid = e.gid
     WHERE v.merged_into IS NULL AND e.gid <> ? AND v.last_seen >= ? AND v.first_seen <= ?
     ORDER BY e.id DESC LIMIT 600'
);
$st->execute([$gid, (float) $v['first_seen'] - $window, (float) $v['last_seen'] + $window]);
// Scored as the matcher scores a track: our mean embedding against each stored sample of the
// other vehicle, averaged over its 3 best samples.
$per = [];
foreach ($st->fetchAll() as $r) {
    if (strlen((string) $r['vec']) !== 4 * count($centre)) {
        continue; // stored by another ReID setup (with/without the camera adapter): not comparable
    }
    $per[$r['gid']][] = $dot($centre, $unpack((string) $r['vec']));
}
$best = [];
foreach ($per as $g => $list) {
    rsort($list);
    $list = array_slice($list, 0, 3);
    $best[$g] = array_sum($list) / count($list);
}
arsort($best);
$top = array_slice($best, 0, 6, true);
if ($top) {
    $in = implode(',', array_fill(0, count($top), '?'));
    $st = $db->prepare("SELECT gid, vclass, color, first_camera, last_camera, thumb_path FROM vehicles WHERE gid IN ($in)");
    $st->execute(array_keys($top));
    $meta = [];
    foreach ($st->fetchAll() as $r) {
        $meta[$r['gid']] = $r;
    }
    foreach ($top as $g => $s) {
        $m = $meta[$g] ?? [];
        $out['nearest'][] = [
            'gid' => $g,
            'similarity' => round($s, 3),
            'type' => $m['vclass'] ?? null,
            'color' => $m['color'] ?? null,
            'camera' => $m['last_camera'] ?? null,
            'thumb' => !empty($m['thumb_path']) ? 'media.php?thumb=' . rawurlencode($g) : null,
        ];
    }
}
App::json($out);
