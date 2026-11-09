<?php
declare(strict_types=1);

require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
$rows = App::db()->query('SELECT id, name, width, height, fps, resolution_weight, status, updated_at FROM cameras ORDER BY id')->fetchAll();
App::json(['items' => $rows]);
