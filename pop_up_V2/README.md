
# Pop-Up Notification System

This project is a Dockerized pop-up notification system for recording thermal sensation votes. It includes a PHP web application for data input, a MySQL database to store the data, and a Python bot (Flask application) for push notifications.

## Project Structure

```plaintext
pop_up_V2/
├── bot/
│   ├── bot_DIGIBUILD.py          # Python bot script (Flask + aiogram for Telegram)
│   ├── Dockerfile                # Dockerfile for Python bot
│   └── requirements.txt          # Python dependencies
├── img/
│   ├── digibuild.png
│   └── tsv.png
├── db.inc.php                    # PHP file for database connection
├── docker-compose.yml            # Docker Compose file to orchestrate services
├── Dockerfile                    # Dockerfile for PHP application
├── index.php                     # Main PHP application file
├── init.sql                      # SQL script to create and populate the database
├── script.js                     # JavaScript for client-side interactions
├── styles.css                    # CSS for styling the PHP application
└── submit.php                    # PHP file to handle form submissions
```

## Prerequisites

- **Docker**: Download and install Docker from [Docker's official website](https://www.docker.com/get-started).
- **Docker Compose**: Docker Compose is typically included with Docker Desktop, but you can also install it separately if necessary. Verify the installation with `docker-compose --version`.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd pop_up_V2
   ```

2. **Build and Start the Docker Containers**:
   Run the following command to build and start the services:
   ```bash
   docker-compose up --build
   ```

   This will start:
   - A MySQL database service (`mysql-db`) on port 3306
   - A PHP web application (`php-app`) on port 8080
   - A Python bot service (`python-bot`) on port 5000

3. **Database Initialization**:
   On first run, the `init.sql` script will automatically create and populate the `digibuild` database with initial data, including tables (`pilots`, `buildings`, `thermal_data`) and sample records.

## Accessing the Application

- **PHP Web Application**:
  Access the web application at [http://localhost:8080](http://localhost:8080). This application displays a form to input thermal sensation votes.

- **Python Bot**:
  The Python bot runs a Flask application on [http://localhost:5000](http://localhost:5000) and will send reminders to a Telegram chat specified in the `CHAT_ID` variable.

## Environment Variables

The following environment variables are used to connect the PHP application to the MySQL database:

- `MYSQL_HOST`: The hostname of the MySQL service (set to `mysql-db`).
- `MYSQL_USER`: The username for MySQL (default is `digibuild`).
- `MYSQL_PASSWORD`: The password for MySQL (default is `DigiBittY23`).
- `MYSQL_DATABASE`: The name of the database (default is `digibuild`).

These values are configured in `docker-compose.yml` and passed to the PHP application automatically.

## Database Schema

The database contains three main tables:

1. **pilots**: Stores information about pilots.
2. **buildings**: Stores information about buildings and links to pilots.
3. **thermal_data**: Stores thermal sensation votes, referencing `pilots` and `buildings`.

Refer to `init.sql` for the exact table structure and initial data.

## Project Components

### 1. PHP Application

- **index.php**: Displays a form for users to submit thermal sensation votes. It retrieves options for "pilot" and "building" from the database.
- **submit.php**: Handles form submissions and stores data in the `thermal_data` table.
- **db.inc.php**: Manages the database connection for PHP using credentials from environment variables.

### 2. Python Bot (Flask Application)

The Python bot sends scheduled reminders to a Telegram chat to encourage users to fill out the thermal sensation form.

- **bot_DIGIBUILD.py**: Contains the Flask server and scheduling logic.
- **requirements.txt**: Lists required Python packages (`Flask` and `aiogram`).

### 3. Docker Configuration

- **Dockerfile**: Dockerfile for the PHP application, based on PHP with Apache.
- **bot/Dockerfile**: Dockerfile for the Python bot.
- **docker-compose.yml**: Defines and links all services (PHP, MySQL, Python).

## Running in Development Mode

If you need to reset the database, you can use:

```bash
docker-compose down -v
docker-compose up --build
```

This will remove all database data and reinitialize it using the `init.sql` script.

## Troubleshooting

- **Port Conflicts**: Ensure that ports 8080 (PHP app) and 5000 (Python bot) are not being used by other services.
- **MySQL Errors**: If you encounter errors with MySQL, try clearing the volume by using the `docker-compose down -v` command to reset the database.

## License
