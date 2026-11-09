<?php
declare(strict_types=1);

// GET api/journey_steps.php?gid=VH-001 - the vehicle's journey as camera visits, in time order,
// with the recorded video and the vehicle's boxes for each visit (dashboard journey replay).
//   steps[]: camera, camera_name, start_ts, end_ts, start_offset, end_offset, clip (media URL|null),
//            boxes [[offset, x1, y1, x2, y2], ...] (0..1), gap_s (time since the previous visit ended)
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
$st = $db->prepare(
    'SELECT s.id, s.camera_id, c.name AS camera_name, s.start_ts, s.end_ts, s.start_offset, s.end_offset,
            s.start_clock, s.end_clock, s.media, s.boxes
     FROM sightings s LEFT JOIN cameras c ON c.id = s.camera_id WHERE s.gid = ? ORDER BY s.start_ts'
);
$st->execute([$gid]);

// Sightings of one camera that touch in time (a track re-linked after an occlusion, two local
// tracks of the same car) form one visit.
$join = (float) App::config('pipeline.journey_join_seconds', 3.0);
$steps = [];
foreach ($st->fetchAll() as $s) {
    $boxes = $s['boxes'] ? (json_decode((string) $s['boxes'], true) ?: []) : [];
    $last = $steps ? count($steps) - 1 : -1;
    // merge into the latest visit of this camera if it continues it
    for ($k = $last; $k >= 0 && $k >= $last - 2; $k--) {
        $p = &$steps[$k];
        if ($p['camera'] === $s['camera_id'] && $p['media'] === $s['media']
            && $s['start_ts'] - $p['end_ts'] <= $join) {
            $p['end_ts'] = max($p['end_ts'], (float) $s['end_ts']);
            $p['start_offset'] = min($p['start_offset'], (float) $s['start_offset']);
            $p['end_offset'] = max($p['end_offset'], (float) $s['end_offset']);
            if ($s['start_clock'] !== null) {
                $p['start_clock'] = $p['start_clock'] === null ? (float) $s['start_clock'] : min($p['start_clock'], (float) $s['start_clock']);
            }
            if ($s['end_clock'] !== null) {
                $p['end_clock'] = max((float) ($p['end_clock'] ?? 0), (float) $s['end_clock']);
            }
            $p['boxes'] = array_merge($p['boxes'], $boxes);
            unset($p);
            continue 2;
        }
        unset($p);
    }
    $steps[] = [
        'sighting' => (int) $s['id'],
        'camera' => $s['camera_id'],
        'camera_name' => $s['camera_name'] ?: $s['camera_id'],
        'start_ts' => (float) $s['start_ts'],
        'end_ts' => (float) $s['end_ts'],
        'start_offset' => (float) $s['start_offset'],
        'end_offset' => (float) $s['end_offset'],
        // shared camera clock (seconds of the day, cameras.clock_start), null for old sightings
        'start_clock' => $s['start_clock'] === null ? null : (float) $s['start_clock'],
        'end_clock' => $s['end_clock'] === null ? null : (float) $s['end_clock'],
        'mask' => App::config('cameras.overrides.' . $s['camera_id'] . '.timestamp_mask', null),
        'media' => $s['media'],
        'boxes' => $boxes,
    ];
}
$pad = (float) App::config('stitching.clip_padding', 1.0);
$maxLen = (float) App::config('stitching.max_clip_seconds', 90);
$out = [];
$prevEnd = null;
foreach ($steps as $p) {
    usort($p['boxes'], fn($a, $b) => $a[0] <=> $b[0]);
    // the clip: the visit plus a little before/after; video time 0 = clip_from in the camera file
    $from = round(max(0.0, $p['start_offset'] - $pad), 1);
    $to = round(min($p['end_offset'] + $pad, $from + $maxLen), 1);
    $p['clip_from'] = $from;
    $p['clip'] = $p['media'] ? sprintf('media.php?sighting=%d&from=%.1f&to=%.1f', $p['sighting'], $from, $to) : null;
    $p['gap_s'] = $prevEnd === null ? null : round($p['start_ts'] - $prevEnd, 1);
    $prevEnd = $prevEnd === null ? $p['end_ts'] : max($prevEnd, $p['end_ts']);
    unset($p['media']);
    $out[] = $p;
}
App::json(['gid' => $gid, 'steps' => $out]);
