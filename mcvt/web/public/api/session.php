<?php
declare(strict_types=1);

require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

$user = App::requireUser(true);
$public = (string) App::config('server.ws_public_url', '');
App::json([
    'user' => ['username' => $user['username'], 'role' => $user['role']],
    'csrf' => App::csrfToken(),
    'ws' => [
        'url' => $public,
        'port' => (int) App::config('server.ws_port', 8090),
        'token' => App::issueWsToken((int) $user['id']),
    ],
    'settings' => [
        'confidence' => App::config('detection.confidence'),
        'reid_threshold' => App::config('reid.similarity_threshold'),
        'hysteresis' => App::config('switching.hysteresis'),
        'buffer_frames' => App::config('switching.buffer_frames'),
        'live_timeout' => App::config('pipeline.vehicle_live_timeout', 3),
    ],
]);
