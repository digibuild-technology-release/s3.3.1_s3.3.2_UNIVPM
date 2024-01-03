<?php

// Database configuration
$hostname = "localhost";
$username = "p2endure";
$password = "termodiism";
$database = "digibuild";

// Create a database connection
$conn = new mysqli($hostname, $username, $password, $database);

// Check the connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}