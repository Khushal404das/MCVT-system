<?php
declare(strict_types=1);

// Start the WebSocket hub:  php web/bin/ws-server.php
if (PHP_SAPI !== 'cli') {
    exit("CLI only\n");
}
require dirname(__DIR__) . '/bootstrap.php';
require dirname(__DIR__) . '/lib/WsServer.php';

use Mcvt\App;
use Mcvt\WsServer;

set_time_limit(0);
$token = App::ensureEngineToken();
App::db(); // make sure the schema exists
$server = new WsServer(
    (string) App::config('server.ws_host', '127.0.0.1'),
    (int) App::config('server.ws_port', 8090),
    $token
);
$server->run();
