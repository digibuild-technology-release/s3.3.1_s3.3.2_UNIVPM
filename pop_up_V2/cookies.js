// Function to get a cookie value by name
function getCookie(name) {
    var value = "; " + document.cookie;
    var parts = value.split("; " + name + "=");
    if (parts.length === 2) return decodeURIComponent(parts.pop().split(";").shift());
}

// Function to set a cookie
function setCookie(name, value, days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + encodeURIComponent(value) + expires + "; path=/";
}

document.addEventListener("DOMContentLoaded", function () {
    // Check if the pilot and building cookies exist
    var pilot = getCookie("pilot");
    var building = getCookie("building");

    // If the pilot cookie exists, pre-fill the form field
    if (pilot) {
        document.getElementById("pilot").value = pilot;
    }

    // If the building cookie exists, pre-fill the form field
    if (building) {
        document.getElementById("building").value = building;
    }

    // ... (similar logic for other fields like floor, and room)

    // Example: Form submission logic
    document.getElementById("your-form-id").addEventListener("submit", function (event) {
        // Prevent the default form submission behavior
        event.preventDefault();

        // Process the form data (insert into the database, etc.)
        // ...

        // Retrieve values from the form
        var submittedPilot = document.getElementById("pilot").value;
        var submittedBuilding = document.getElementById("building").value;

        // Set or update the pilot and building cookies
        setCookie("pilot", submittedPilot, 365); // Set the cookie to last for 365 days
        setCookie("building", submittedBuilding, 365); // Set the cookie to last for 365 days

        // Reset the form or hide the popup after successful submission
        document.getElementById("your-form-id").reset(); // Reset the form
    });
});
