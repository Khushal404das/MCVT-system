<?php
declare(strict_types=1);

require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
$limit = max(1, min(500, (int) ($_GET['limit'] ?? 100)));
$rows = App::db()->query("SELECT ts, level, source, message FROM logs ORDER BY id DESC LIMIT $limit")->fetchAll();
App::json(['items' => array_reverse($rows)]);
