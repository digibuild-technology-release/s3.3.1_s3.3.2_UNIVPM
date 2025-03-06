<?php

	include("db.inc.php");

// Set the timezone
date_default_timezone_set('Europe/Rome'); // Replace 'Your/Timezone' with your timezone identifier


	
// Process the form data
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Get form data
    $room = $_POST["room"];
    $floor = $_POST["floor"];
    $building = $_POST["building"];
    $pilot = $_POST["pilot"];
    $vote = $_POST["vote"];


// Query to retrieve the corresponding name from the lookup table
	$pilotLookupQuery = "SELECT `name` FROM `pilots` WHERE `id` = $pilot";
	$pilotResult = $conn->query($pilotLookupQuery);
    $pilotRow = $pilotResult->fetch_assoc();
    $pilotName = $pilotRow["name"];

	$buildingLookupQuery = "SELECT `name` FROM `buildings` WHERE `id` = $building";
	$buildingResult = $conn->query($buildingLookupQuery);
    $buildingRow = $buildingResult->fetch_assoc();
    $buildingName = $buildingRow["name"];


    // Get the current date and time
    $date = date("Y-m-d H:i:s");

    $tableName = "thermal_data"; 

    // Insert data into the database
    $sql = "INSERT INTO $tableName (date, TSV, room, floor, building_id, pilot, building) VALUES ('$date', '$vote', '$room', '$floor', '$building', '$pilotName', '$buildingName')";


    if ($conn->query($sql) === TRUE) {
        echo "Record inserted successfully";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

// Close the database connection
$conn->close();

?>
