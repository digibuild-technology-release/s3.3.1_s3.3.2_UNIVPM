-- Create the database
CREATE DATABASE IF NOT EXISTS digibuild;
USE digibuild;

-- Create the pilots table
CREATE TABLE IF NOT EXISTS pilots (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    label VARCHAR(100) NOT NULL
) ENGINE=InnoDB;

-- Create the buildings table
CREATE TABLE IF NOT EXISTS buildings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    pilot_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    label VARCHAR(100) NOT NULL,
    FOREIGN KEY (pilot_id) REFERENCES pilots(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Create the thermal_data table
CREATE TABLE IF NOT EXISTS thermal_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    dates DATETIME NOT NULL,
    pilot VARCHAR(100) NOT NULL,
    building VARCHAR(100) NOT NULL,
    building_id INT NOT NULL,
    floor INT NOT NULL,
    room VARCHAR(100) NOT NULL,
    TSV INT NOT NULL,
    FOREIGN KEY (building_id) REFERENCES buildings(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Populate the pilots table
INSERT INTO pilots (id, name, label) VALUES
(1, 'UNIVPM', 'UNIVPM'),
(2, 'FOCCHI', 'FOCCHI'),
(3, 'Helsinki', 'Helsinki'),
(4, 'IECCP', 'IECCP'),
(5, 'UCL', 'UCL'),
(6, 'EDF', 'EDF');

-- Populate the buildings table
INSERT INTO buildings (id, pilot_id, name, label) VALUES
(1, 1, 'Faculty of Engineering', 'UNIVPM - Faculty of Engineering'),
(2, 2, 'Headquarters', 'Focchi - Headquarters'),
(3, 3, 'KYMP Building', 'Helsinki - KYMP Building'),
(4, 4, 'Office Building', 'IECCP - Office Building'),
(5, 4, 'Residential Building', 'IECCP - Residential Building'),
(6, 5, 'Marshgate', 'UCL - Marshgate'),
(7, 5, 'One Pool Street', 'UCL - One Pool Street'),
(8, 6, 'Headquarters', 'EDF - Headquarters');

