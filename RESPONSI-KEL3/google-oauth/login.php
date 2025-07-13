<?php
session_start();

// Konfigurasi Google OAuth
$client_id = "1001049367442-uvo9nfj2jsjvj8qi8a7mopu940u26lu0.apps.googleusercontent.com";
$redirect_uri = "http://localhost/google-oauth/callback.php";
$scope = "email profile";
$response_type = "code";

// Generate State Token untuk CSRF Protection (Kompatibel PHP 5.6)
$state = bin2hex(openssl_random_pseudo_bytes(16));
$_SESSION['oauth_state'] = $state;

// Buat URL Authorization Google dengan prompt select_account
$auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" . http_build_query([
    'client_id' => $client_id,
    'redirect_uri' => $redirect_uri,
    'response_type' => $response_type,
    'scope' => $scope,
    'state' => $state,
    'prompt' => 'select_account' // Memunculkan pilihan akun setiap login
]);

// Optional: Tampilkan error dari callback
if (isset($_GET['error'])) {
    $error_message = '';
    switch ($_GET['error']) {
        case 'csrf':
            $error_message = 'Token CSRF tidak valid.';
            break;
        case 'code_missing':
            $error_message = 'Kode otorisasi tidak tersedia.';
            break;
        case 'token_request_failed':
            $error_message = 'Gagal terhubung ke server Google.';
            break;
        case 'invalid_token':
            $error_message = 'Gagal mendapatkan access token.';
            break;
        case 'userinfo_failed':
            $error_message = 'Gagal mendapatkan data user.';
            break;
        default:
            $error_message = 'Terjadi kesalahan yang tidak diketahui.';
    }
    echo "<script>alert('$error_message');</script>";
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login Page</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, rgb(128, 196, 255),rgb(236, 187, 231));
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            overflow: hidden;
        }
        /* Background Bubble */
        .bubble {
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.3);
            z-index: 0;
            animation: float 15s infinite;
        }
        .bubble:nth-child(1) { width: 100px; height: 100px; top: 10%; left: 15%; }
        .bubble:nth-child(2) { width: 150px; height: 150px; top: 30%; left: 75%; }
        .bubble:nth-child(3) { width: 80px; height: 80px; top: 65%; left: 20%; }
        .bubble:nth-child(4) { width: 120px; height: 120px; top: 80%; left: 70%; }
        .bubble:nth-child(5) { width: 50px; height: 50px; top: 50%; left: 45%; }

        @keyframes float {
            0% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0); }
        }

        .login-container {
            background: white;
            padding: 40px;
            border-radius: 25px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            text-align: center;
            max-width: 300px;
            width: 90%;
            z-index: 1;
        }

        h2 {
            margin-bottom: 25px;
            color: #333;
        }

        .google-btn {
            display: inline-flex;
            align-items: center;
            background-color: #4285F4;
            color: white;
            padding: 12px 70px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 16px;
            transition: background-color 0.3s ease;
            margin-top: 20px;
        }

        .google-btn:hover {
            background-color: #2c3e50;
        }

        .google-btn img {
            width: 20px;
            margin-right: 10px;
        }

        form {
            display: flex;
            flex-direction: column;
        }

        input[type="text"],
        input[type="password"] {
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        button {
            background-color: #4caf50;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #45a049;
        }

        .divider {
            margin: 10px 0;
            font-weight: bold;
            color: #888;
        }
    </style>
</head>
<body>

    <!-- Background Bubbles -->
    <div class="bubble"></div>
    <div class="bubble"></div>
    <div class="bubble"></div>
    <div class="bubble"></div>
    <div class="bubble"></div>

    <!-- Login Container -->
    <div class="login-container">
        <h2>Login</h2>

        <!-- Manual Login -->
        <form action="login_process.php" method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login With Username</button>
        </form>

        <div class="divider">OR</div>

        <!-- Login with Google -->
        <a class="google-btn" href="<?php echo htmlspecialchars($auth_url); ?>">
            <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google Logo">
            Login with Google
        </a>
    </div>

</body>
</html>
