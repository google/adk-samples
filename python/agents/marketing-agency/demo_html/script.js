document.addEventListener('DOMContentLoaded', function() {

    // --- Mobile Navigation Toggle ---
    const navToggle = document.querySelector('.nav-toggle');
    const mainNav = document.querySelector('.main-nav');

    if (navToggle && mainNav) {
        navToggle.addEventListener('click', function() {
            const isExpanded = this.getAttribute('aria-expanded') === 'true';
            this.setAttribute('aria-expanded', !isExpanded);
            mainNav.classList.toggle('is-open');
        });
    }

    // --- Dynamic Year in Footer ---
    const yearSpan = document.getElementById('current-year');
    if (yearSpan) {
        yearSpan.textContent = new Date().getFullYear();
    }

    // --- Basic Form Status Placeholder (Optional) ---
    // This is a very basic example. Real form handling needs more robust JS and backend.
    const contactForm = document.getElementById('contact-form');
    const formStatus = document.getElementById('form-status');

    if (contactForm && formStatus) {
        contactForm.addEventListener('submit', function(e) {
            // Prevent actual submission for this example
            e.preventDefault();

            // Simulate form submission feedback
            formStatus.textContent = 'Sending message...'; // Replace with actual success/error handling
            formStatus.style.color = 'orange';

            // Simulate a response (replace with actual AJAX call result)
            setTimeout(() => {
                // On success:
                // formStatus.textContent = 'Message sent successfully!';
                // formStatus.style.color = 'green';
                // contactForm.reset(); // Clear the form

                // On error (or for this placeholder):
                 formStatus.textContent = 'Form submission simulated. Setup required for actual sending.';
                 formStatus.style.color = 'red';
            }, 1500);
        });
    }

     // --- Placeholder for 'Add to Cart' Button Click ---
     const addToCartButtons = document.querySelectorAll('.add-to-cart-btn');
     addToCartButtons.forEach(button => {
         button.addEventListener('click', () => {
             alert('Placeholder: "Add to Cart" functionality requires e-commerce backend setup.');
             // In a real implementation, this would trigger adding the item to a cart object/state.
         });
     });

});