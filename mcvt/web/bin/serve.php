<?php
declare(strict_types=1);

// Start the PHP web server on the host/port from config.yaml:
//   php web/bin/serve.php
if (PHP_SAPI !== 'cli') {
    exit("CLI only\n");
}
require dirname(__DIR__) . '/bootstrap.php';

use Mcvt\App;

$host = (string) App::config('server.web_host', '127.0.0.1');
$port = (int) App::config('server.web_port', 8080);
$docroot = dirname(__DIR__) . '/public';
echo "MCVT web UI on http://$host:$port\n";
passthru(escapeshellarg(PHP_BINARY) . ' -S ' . escapeshellarg("$host:$port") . ' -t ' . escapeshellarg($docroot));
