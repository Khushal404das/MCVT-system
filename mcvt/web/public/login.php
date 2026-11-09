<?php
declare(strict_types=1);

require dirname(__DIR__) . '/bootstrap.php';

use Mcvt\App;

App::startSession();
$error = '';
$noUsers = (int) App::db()->query('SELECT COUNT(*) FROM users')->fetchColumn() === 0;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $u = trim((string) ($_POST['username'] ?? ''));
    $p = (string) ($_POST['password'] ?? '');
    if (App::login($u, $p)) {
        header('Location: index.php');
        exit;
    }
    usleep(400000); // slow down guessing
    $error = 'Wrong username or password.';
}
if (App::user()) {
    header('Location: index.php');
    exit;
}
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCVT · Sign in</title>
<link rel="stylesheet" href="assets/app.css">
</head>
<body class="login-body">
<main class="login-card">
  <div class="brand brand-lg"><span class="brand-mark"></span><div><strong>MCVT</strong><small>Multi-Camera Vehicle Tracking</small></div></div>
  <?php if ($noUsers): ?>
    <p class="notice">No operator account exists yet. Create one from a terminal in the project folder:<br>
      <code>php web/bin/create-user.php &lt;username&gt; admin</code></p>
  <?php endif; ?>
  <form method="post" autocomplete="on">
    <label>Username<input name="username" required autofocus autocomplete="username"></label>
    <label>Password<input name="password" type="password" required autocomplete="current-password"></label>
    <?php if ($error): ?><p class="form-error" role="alert"><?= App::e($error) ?></p><?php endif; ?>
    <button class="btn btn-primary" type="submit">Sign in</button>
  </form>
</main>
</body>
</html>
