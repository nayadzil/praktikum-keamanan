<?php
session_start();
include 'config.php'; // Memuat $client_id, $client_secret, $redirect_uri

// 1. Verifikasi nilai state untuk CSRF Protection
if (!isset($_GET['state']) || !isset($_SESSION['oauth_state']) || $_GET['state'] !== $_SESSION['oauth_state']) {
    header('Location: login.php?error=csrf');
    exit();
}

// 2. Tangkap code dari URL
if (!isset($_GET['code'])) {
    header('Location: login.php?error=code_missing');
    exit();
}

$code = $_GET['code'];

// 3. Kirim HTTP POST ke Google Token Endpoint
$token_url = 'https://oauth2.googleapis.com/token';
$token_data = [
    'code' => $code,
    'client_id' => $client_id,
    'client_secret' => $client_secret,
    'redirect_uri' => $redirect_uri,
    'grant_type' => 'authorization_code'
];

$curl = curl_init();
curl_setopt_array($curl, [
    CURLOPT_URL => $token_url,
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => http_build_query($token_data),
    CURLOPT_RETURNTRANSFER => true,
]);

$token_response = curl_exec($curl);

if (curl_errno($curl)) {
    header('Location: login.php?error=token_request_failed');
    exit();
}

curl_close($curl);

$token_json = json_decode($token_response, true);

// 4. Ambil access_token dari response
if (!isset($token_json['access_token'])) {
    header('Location: login.php?error=invalid_token');
    exit();
}

$access_token = $token_json['access_token'];

// 5. Gunakan access_token untuk request data profil user
$userinfo_url = 'https://www.googleapis.com/oauth2/v1/userinfo?access_token=' . urlencode($access_token);

$curl = curl_init();
curl_setopt_array($curl, [
    CURLOPT_URL => $userinfo_url,
    CURLOPT_RETURNTRANSFER => true,
]);

$userinfo_response = curl_exec($curl);

if (curl_errno($curl)) {
    header('Location: login.php?error=userinfo_failed');
    exit();
}

curl_close($curl);

$user = json_decode($userinfo_response, true);

// 6. Simpan data user ke session
$_SESSION['user_name'] = isset($user['name']) ? $user['name'] : 'Unknown';
$_SESSION['user_email'] = isset($user['email']) ? $user['email'] : 'Unknown';

// 7. Redirect ke halaman dashboard
header('Location: dashboard.php');
exit();
?>
