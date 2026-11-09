<?php
declare(strict_types=1);

// Authenticated access to stored media (NFR-S-03). Files live outside the web root.
//   media.php?thumb=VH-001          vehicle thumbnail
//   media.php?journey=12[&download] stitched journey video (supports HTTP Range for seeking)
//   media.php?sighting=345&from=12.5&to=24   a short clip of the camera video that sighting was
//                                    analysed from (journey replay in the dashboard)
require dirname(__DIR__) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
session_write_close();
$storage = realpath(App::path('paths.storage', 'storage'));
$file = null;
$type = 'application/octet-stream';

if (isset($_GET['thumb']) && preg_match('/^[A-Za-z0-9_-]{1,32}$/', (string) $_GET['thumb'])) {
    $file = $storage . DIRECTORY_SEPARATOR . 'thumbs' . DIRECTORY_SEPARATOR . $_GET['thumb'] . '.jpg';
    $type = 'image/jpeg';
} elseif (isset($_GET['journey'])) {
    $st = App::db()->prepare('SELECT output_path FROM jobs WHERE id = ?');
    $st->execute([(int) $_GET['journey']]);
    $rel = $st->fetchColumn();
    if ($rel) {
        $file = $storage . DIRECTORY_SEPARATOR . str_replace(['/', '\\'], DIRECTORY_SEPARATOR, (string) $rel);
        $type = 'video/mp4';
    }
} elseif (isset($_GET['sighting'], $_GET['from'], $_GET['to'])) {
    // A short clip [from, to] (seconds) of the video a sighting was analysed from. Cut once with
    // ffmpeg and cached: the PHP built-in server answers one request at a time, so it must only
    // ever send small files (a browser that stops reading a big video would block every request).
    $from = round((float) $_GET['from'], 1);
    $to = round((float) $_GET['to'], 1);
    $maxLen = (float) App::config('stitching.max_clip_seconds', 90);
    $st = App::db()->prepare('SELECT media, camera_id FROM sightings WHERE id = ?');
    $st->execute([(int) $_GET['sighting']]);
    $row = $st->fetch() ?: ['media' => '', 'camera_id' => ''];
    $media = str_replace('\\', '/', (string) $row['media']);
    // the camera's own burned-in timestamp is covered (cameras.overrides.<id>.timestamp_mask);
    // the dashboard draws the shared clock in its place
    $mask = App::config('cameras.overrides.' . $row['camera_id'] . '.timestamp_mask', null);
    $vf = '';
    if (is_array($mask) && count($mask) === 4) {
        [$mx1, $my1, $mx2, $my2] = array_map('floatval', $mask);
        $vf = sprintf(' -vf "drawbox=x=iw*%.4f:y=ih*%.4f:w=iw*%.4f:h=ih*%.4f:color=0x121212:t=fill"',
            $mx1, $my1, $mx2 - $mx1, $my2 - $my1);
    }
    $root = str_replace('\\', '/', dirname(__DIR__, 2));
    $src = $media === '' ? '' : (preg_match('#^([A-Za-z]:/|/)#', $media) ? $media : "$root/$media");
    $srcReal = $src !== '' ? realpath($src) : false;
    $ffmpegFile = $storage . DIRECTORY_SEPARATOR . 'ffmpeg_path.txt';
    $ffmpeg = (string) App::config('stitching.ffmpeg', 'auto');
    if ($ffmpeg === 'auto' && is_file($ffmpegFile)) {
        $ffmpeg = trim((string) file_get_contents($ffmpegFile)); // written by the engine at start
    }
    if ($srcReal && $from >= 0 && $to > $from && $to - $from <= $maxLen + 0.5 && $ffmpeg !== 'auto' && is_file($ffmpeg)) {
        $dir = $storage . DIRECTORY_SEPARATOR . 'journey_clips';
        if (!is_dir($dir)) {
            mkdir($dir, 0770, true);
        }
        $out = $dir . DIRECTORY_SEPARATOR . sprintf('%d_%d_%d%s.mp4', (int) $_GET['sighting'], (int) round($from * 10),
            (int) round($to * 10), $vf !== '' ? '_m' : '');
        if (!is_file($out)) {
            $tmp = $out . '.tmp.mp4';
            $cmd = escapeshellarg($ffmpeg) . ' -y -loglevel error -ss ' . sprintf('%.2f', $from) . ' -i ' . escapeshellarg($srcReal)
                . ' -t ' . sprintf('%.2f', $to - $from) . $vf . ' -an -c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p'
                . ' -movflags +faststart ' . escapeshellarg($tmp) . ' 2>&1';
            exec($cmd, $lines, $code);
            if ($code === 0 && is_file($tmp) && filesize($tmp) > 0) {
                rename($tmp, $out);
            } else {
                @unlink($tmp);
                error_log('journey clip failed: ' . implode(' ', array_slice($lines, -3)));
            }
        }
        $file = is_file($out) ? $out : null;
        $type = 'video/mp4';
    }
}
$real = $file ? realpath($file) : false;
if (!$real || !$storage || !str_starts_with($real, $storage . DIRECTORY_SEPARATOR) || !is_file($real)) {
    http_response_code(404);
    exit;
}

$size = filesize($real);
$start = 0;
$end = $size - 1;
header('Content-Type: ' . $type);
header('Accept-Ranges: bytes');
header('Cache-Control: private, max-age=' . ($type === 'image/jpeg' ? 5 : 3600));
if (isset($_GET['download'])) {
    header('Content-Disposition: attachment; filename="' . basename($real) . '"');
}
if (isset($_SERVER['HTTP_RANGE']) && preg_match('/bytes=(\d*)-(\d*)/', $_SERVER['HTTP_RANGE'], $m)) {
    if ($m[1] === '' && $m[2] !== '') {
        $start = max(0, $size - (int) $m[2]);
    } else {
        $start = (int) $m[1];
        if ($m[2] !== '') {
            $end = min($end, (int) $m[2]);
        }
    }
    if ($start > $end || $start >= $size) {
        http_response_code(416);
        header("Content-Range: bytes */$size");
        exit;
    }
    http_response_code(206);
    header("Content-Range: bytes $start-$end/$size");
}
header('Content-Length: ' . ($end - $start + 1));
$fh = fopen($real, 'rb');
fseek($fh, $start);
$left = $end - $start + 1;
while ($left > 0 && !feof($fh)) {
    $chunk = fread($fh, (int) min(262144, $left));
    echo $chunk;
    $left -= strlen($chunk);
    flush();
}
fclose($fh);
