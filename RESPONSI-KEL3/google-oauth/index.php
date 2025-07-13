<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Implementasi Google OAuth2 dengan PHP Native</title>
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #7f7fd5, #86a8e7, #91eae4);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        .container {
            background: white;
            border-radius: 20px;
            width: 90%;
            max-width: 850px;
            padding: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 10px rgba(0,0,0,0.2);
            position: relative;
        }

        .content {
            max-width: 500px;
        }

        h1 {
            font-size: 30px;
            color: #4b4bff;
            margin-bottom: 20px;
        }

        p {
            font-size: 16px;
            color: #666;
            margin-bottom: 30px;
        }

        .buttons {
            display: flex;
            gap: 15px;
        }

        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            transition: 0.3s;
        }

        .btn-primary {
            background-color: #4b4bff;
            color: white;
        }

        .btn-primary:hover {
            background-color: #2c3e50;
        }

        .btn-secondary {
            background-color: transparent;
            color: #4b4bff;
            border: 2px solid #4b4bff;
        }

        .btn-secondary:hover {
            background-color: #4b4bff;
            color: white;
        }

        .image-section img {
            width: 250px;
        }

        /* Dekorasi Lingkaran */
        .circle {
            position: absolute;
            border-radius: 50%;
            opacity: 0.7;
            z-index: -1;
        }

        .circle1 {
            width: 200px;
            height: 200px;
            background: linear-gradient(135deg, #7f7fd5, #86a8e7);
            top: -50px;
            left: -50px;
        }

        .circle2 {
            width: 150px;
            height: 150px;
            background: linear-gradient(135deg, #91eae4, #7f7fd5);
            bottom: -40px;
            right: -40px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                text-align: center;
            }
            .image-section img {
                width: 250px;
                margin-top: 20px;
            }
        }
    </style>
</head>
<body>

    <div class="circle circle1"></div>
    <div class="circle circle2"></div>

    <div class="container">
        <div class="content">
            <h1>Final Project Keamanan Sistem Informasi & Jaringan</h1>
            <p>Silakan login untuk melanjutkan.</p>
            <div class="buttons">
                <a href="login.php" class="btn btn-primary">Login Sekarang</a>
            </div>
        </div>
        <div class="image-section">
            <img src="laptop.jpg" alt="Ultrabook Illustration">
        </div>
    </div>

</body>
</html>
