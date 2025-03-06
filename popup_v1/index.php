<?php

	include("db.inc.php");
	
?><!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thermal Sensation Vote</title>
    <link rel="stylesheet" href="styles.css">
	
	<script src="script.js"></script>
	
    <script src="cookies.js"></script> <!-- Include the JavaScript file -->

	
</head>
<body>

<div id="popup-container">
    <div id="popup">
        <span id="close-popup">&times;</span>
        <h2>Thermal Sensation Vote</h2>
        <form id="your-form-id" action="submit.php" method="post">
 
 			<label for="pilot">Pilot:</label>
            <select name="pilot" required>
                <option value="" style="color: #666;">&rarr; Select pilot &larr;</option>
<?php
    // Insert data into the database
    $sql = 'SELECT * FROM `pilots`';
	if ($result = $conn->query($sql)) {
		while($row = $result->fetch_array()){
			echo '                <option value="'.$row['id'].'">'.$row['label'].'</option>';
		}
	}
?>
            </select>
 

 
            <label for="building">Building:</label>
            <select name="building" required>
                <option value="" style="color: #666;">&rarr; Select building &larr;</option>

<?php
    // Insert data into the database
    $sql = 'SELECT * FROM `buildings`';
	if ($result = $conn->query($sql)) {
		while($row = $result->fetch_array()){
			echo '                <option value="'.$row['id'].'">'.$row['label'].'</option>';
		}
	}
?>
            </select>
			
			<label for="room">Room_ID:</label>
            <input type="text" name="room" required>

            <label for="floor">Floor:</label>
            <select name="floor" required>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
                <option value="9">9</option>
                <option value="10">10</option>
            </select>



            <label for="vote">Thermal Sensation Vote:</label>
            <select name="vote" required>
                <option value="-3">-3</option>
                <option value="-2">-2</option>
                <option value="-1">-1</option>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
            </select>

            <button type="submit">Submit</button>
        </form>
    </div>
</div>



</body>
</html>
