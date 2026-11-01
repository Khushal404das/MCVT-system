<?php
/**
 * MJPEG Stream Proxy
 * Forwards camera streams from Flask backend (port 5000) through PHP (port 8080)
 * to avoid cross-origin issues with MJPEG in <img> tags.
 */

$API_BASE = 'http://localhost:5000';

$camera_id = isset($_GET['cam']) ? preg_replace('/[^a-z0-9]/', '', $_GET['cam']) : '';
$type      = isset($_GET['type']) ? $_GET['type'] : 'thumbnail'; // 'stream' or 'thumbnail'

if (empty($camera_id)) {
    http_response_code(400);
    echo 'Missing cam parameter';
    exit;
}

$endpoint = ($type === 'stream') ? "/stream/{$camera_id}" : "/thumbnail/{$camera_id}";
$url      = $API_BASE . $endpoint;

// Open the upstream MJPEG stream
$ctx = stream_context_create([
    'http' => [
        'timeout'  => 10,
        'method'   => 'GET',
    ]
]);

$stream = @fopen($url, 'rb', false, $ctx);
if (!$stream) {
    http_response_code(502);
    echo 'Cannot connect to backend stream';
    exit;
}

// Forward content-type header
$meta = stream_get_meta_data($stream);
$content_type = 'multipart/x-mixed-replace; boundary=frame';
foreach (($meta['wrapper_data'] ?? []) as $header) {
    if (stripos($header, 'Content-Type:') === 0) {
        $content_type = trim(substr($header, 13));
        break;
    }
}

header('Content-Type: ' . $content_type);
header('Cache-Control: no-cache, no-store');
header('Access-Control-Allow-Origin: *');

// Stream bytes to client
set_time_limit(0);
while (!feof($stream) && !connection_aborted()) {
    $chunk = fread($stream, 8192);
    if ($chunk === false) break;
    echo $chunk;
    flush();
}

fclose($stream);
