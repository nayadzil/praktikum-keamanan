<?php
session_start();
if (!isset($_SESSION['user_name'])) {
    header('Location: login.php');
    exit();
}

// Ambil data user dari session
$name = $_SESSION['user_name'];
$email = $_SESSION['user_email'];
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Logout Confirmation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, rgb(128, 196, 255), rgb(229, 161, 236));
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            overflow: hidden;
        }

        .circle {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.15);
            animation: float 10s infinite ease-in-out alternate;
        }

        .circle:nth-child(1) { width: 100px; height: 100px; top: 10%; left: 20%; }
        .circle:nth-child(2) { width: 150px; height: 150px; top: 60%; left: 70%; }
        .circle:nth-child(3) { width: 80px; height: 80px; top: 30%; left: 80%; }
        .circle:nth-child(4) { width: 120px; height: 120px; top: 70%; left: 15%; }

        @keyframes float {
            0% { transform: translateY(0px); }
            100% { transform: translateY(-30px); }
        }

        .logout-container {
            background: white;
            padding: 15px;
            border-radius: 20px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            text-align: center;
            max-width: 280px;
            width: 100%;
            z-index: 1;
        }

        h2 {
            margin-bottom: 5px;
            color: #333;
            font-size: 18px;
        }

        p {
            margin-bottom: 10px;
            color: #555;
            font-size: 14px;
        }

        .btn {
            display: inline-block;
            padding: 8px 15px;
            margin: 0 5px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .btn-logout {
            background-color: #e74c3c;
            color: white;
        }

        .btn-logout:hover {
            background-color: #c0392b;
        }

        .btn-cancel {
            background-color: #3498db;
            color: white;
        }

        .btn-cancel:hover {
            background-color: #2980b9;
        }

        .email-icon {
            font-size: 18px;
            margin-right: 5px;
            color: #555;
        }
    </style>
</head>

<body>

    <!-- Background Circles -->
    <div class="circle"></div>
    <div class="circle"></div>
    <div class="circle"></div>
    <div class="circle"></div>

    <div class="logout-container">
        <h2><?php echo htmlspecialchars($name); ?></h2>
        <p><span class="email-icon">📧</span><?php echo htmlspecialchars($email); ?></p>
        <p>Apakah Anda yakin ingin logout?</p>

        <a class="btn btn-logout" href="index.php">Ya, Logout</a>
        <a class="btn btn-cancel" href="dashboard.php">Batal</a>
    </div>

</body>

</html>
