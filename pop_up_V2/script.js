document.addEventListener("DOMContentLoaded", function () {
    // Function to show the reminder message
    function showReminderMessage() {
        alert("Please submit your vote!");
    }

    // Set an interval to show the reminder message every 2 minutes
    setInterval(showReminderMessage, 3600000); // 60 minutes in milliseconds

    // Function to handle form submission with AJAX
    function submitFormWithAjax() {
        // Serialize form data
        var formData = new FormData(document.getElementById("your-form-id"));

        // Send an AJAX request for asynchronous form submission
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "submit.php", true);
        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {

                    // Optionally, display a message or perform additional actions
                    alert("Thank you for your vote!");

                    // Show the popup again after submission
                    showPopup();
                } else {
                    // Handle errors or show an error message to the user
                    console.error("Form submission failed. Status: " + xhr.status);
                    // Optionally, display an error message to the user
                    alert("Form submission failed. Please try again.");
                }
            }
        };
        xhr.send(formData);
    }

    // Add a click event listener to the close button
    document.getElementById("close-popup").addEventListener("click", function () {
        // Close the popup by setting its display to "none"
        document.getElementById("popup-container").style.display = "none";
    });

    // Check if the pilot cookie exists
    var pilot = getCookie("pilot");

    // If the pilot cookie exists, pre-fill the form field
    if (pilot) {
        document.getElementById("pilot").value = pilot;
    }

    // ... (similar logic for other fields like building, floor, and room)

    // Example: Form submission logic using AJAX
    document.getElementById("your-form-id").addEventListener("submit", function (event) {
        // Prevent the default form submission behavior
        event.preventDefault();

        // Call the function to handle form submission with AJAX
        submitFormWithAjax();
    });
});
