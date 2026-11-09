<?php
declare(strict_types=1);

require dirname(__DIR__, 2) . '/bootstrap.php';

use Mcvt\App;

App::requireUser(true);
$rows = App::db()->query(
    'SELECT id, gid, recipient, status, message, output_path, created_at, updated_at FROM jobs ORDER BY id DESC LIMIT 50'
)->fetchAll();
foreach ($rows as &$r) {
    $r['video'] = $r['output_path'] ? 'media.php?journey=' . (int) $r['id'] : null;
    unset($r['output_path']);
}
App::json(['items' => $rows]);
