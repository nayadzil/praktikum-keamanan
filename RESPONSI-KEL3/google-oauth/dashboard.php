<?php
session_start();
if (!isset($_SESSION['user_name'])) {
    header('Location: login.php');
    exit();
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dashboard - Project Kelompok 3</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, rgb(128, 196, 255), rgb(229, 161, 236));
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 40px 20px;
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

        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            width: 100%;
            text-align: center;
            z-index: 1;
        }

        .welcome {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .email {
            font-size: 16px;
            margin-bottom: 30px;
        }

        h3 {
            margin-bottom: 20px;
            color: #0059b3;
            font-size: 20px;
            font-weight: bold;
        }

        .bubble-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            justify-content: center;
            margin-bottom: 30px;
        }

        .bubble-item {
            border: 2px solid;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        /* Warna kotak masing-masing */
        .bubble-item:nth-child(1) { border-color: #ff6f61; }
        .bubble-item:nth-child(2) { border-color: #6fcf97; }
        .bubble-item:nth-child(3) { border-color: #f4d35e; }
        .bubble-item:nth-child(4) { border-color: #00bcd4; }

        .bubble-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }

        .bubble {
            background-color: #0059b3;
            border-radius: 50%;
            width: 100px;
            height: 100px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
        }

        .bubble img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%;
        }

        .bubble-item h4 {
            font-size: 16px;
            font-weight: bold;
            color: #0059b3;
            text-align: center;
            margin-bottom: 5px;
        }

        .bubble-item p {
            font-size: 13px;
            color: rgb(9, 9, 10);
            text-align: center;
            margin: 0 0 5px 0;
        }

        .github-btn {
            display: inline-block;
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 12px;
            margin-top: 5px;
            transition: background-color 0.3s ease;
        }

        .github-btn:hover {
            background-color: #555;
        }

        .logout-btn {
            display: inline-block;
            background-color: #e74c3c;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 16px;
            margin-top: 25px;
            transition: background-color 0.3s ease;
        }

        .logout-btn:hover {
            background-color: #c0392b;
        }
    </style>
</head>

<body>

    <!-- Background Circles -->
    <div class="circle"></div>
    <div class="circle"></div>
    <div class="circle"></div>
    <div class="circle"></div>

    <div class="container">
        <div class="welcome">Selamat Datang, <?php echo htmlspecialchars($_SESSION['user_name']); ?>!</div>
        <div class="email">Email: <?php echo htmlspecialchars($_SESSION['user_email']); ?></div>

        <h3>Tim Kolaborasi Project Kelompok 3</h3>
        <div class="bubble-container">
            <div class="bubble-item">
                <div class="bubble">
                    <img src="ali.jpg" alt="Alpridel Jimnoris">
                </div>
                <h4>Alpridel Jimnoris</h4>
                <p>22230003</p>
                <a class="github-btn" href="https://github.com/alprideljim" target="_blank">🔗 GitHub</a>
            </div>
            <div class="bubble-item">
                <div class="bubble">
                    <img src="uput.jpg" alt="Puput Lestari">
                </div>
                <h4>Puput Lestari</h4>
                <p>22230005</p>
                <a class="github-btn" href="https://github.com/" target="_blank">🔗 GitHub</a>
            </div>
            <div class="bubble-item">
                <div class="bubble">
                    <img src="naya.jpg" alt="Inayah Dzil">
                </div>
                <h4>Inayah Dzil</h4>
                <p>22230012</p>
                <a class="github-btn" href="https://github.com/inayahdzil" target="_blank">🔗 GitHub</a>
            </div>
            <div class="bubble-item">
                <div class="bubble">
                    <img src="dimas.jpg" alt="Dimas Oktavian">
                </div>
                <h4>Dimas Oktavian</h4>
                <p>22230016</p>
                <a class="github-btn" href="https://github.com/dimasokta" target="_blank">🔗 GitHub</a>
            </div>
        </div>

        <a class="logout-btn" href="logout.php">Logout</a>
    </div>

</body>
</html>
