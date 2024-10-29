# Use a PHP image with Apache
FROM php:8.0-apache

# Install necessary PHP extensions
RUN docker-php-ext-install mysqli

# Copy the PHP code to the Apache server directory
COPY . /var/www/html/

# Set working directory
WORKDIR /var/www/html

# Expose port 80 for web access
EXPOSE 80
