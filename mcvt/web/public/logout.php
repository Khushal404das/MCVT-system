<?php
declare(strict_types=1);

require dirname(__DIR__) . '/bootstrap.php';

use Mcvt\App;

App::startSession();
$_SESSION = [];
session_destroy();
header('Location: login.php');
