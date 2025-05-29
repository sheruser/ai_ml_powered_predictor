document.addEventListener('DOMContentLoaded', function () {
    // Update slider values as they change
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const output = slider.nextElementSibling;

        // Set initial value
        if (output && output.classList.contains('slider-value')) {
            output.textContent = slider.value;
        }

        // Update value on change
        slider.addEventListener('input', function () {
            if (output) {
                output.textContent = this.value;
            }
        });
    });

    // Collapsible functionality
    const collapsibles = document.querySelectorAll('.collapsible-header');
    collapsibles.forEach(header => {
        header.addEventListener('click', function () {
            const content = this.nextElementSibling;
            content.classList.toggle('active');

            // Change arrow icon
            const icon = this.querySelector('.fa-chevron-down, .fa-chevron-up');
            if (icon) {
                icon.classList.toggle('fa-chevron-down');
                icon.classList.toggle('fa-chevron-up');
            }
        });
    });

    // Form validation
    const form = document.getElementById('personality-form');
    if (form) {
        form.addEventListener('submit', function (e) {
            // You could add validation here if needed
            // e.preventDefault(); // Uncomment to prevent form submission for validation

            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
                submitBtn.disabled = true;
            }
        });
    }

    // Add animation to result card if present
    const resultCard = document.querySelector('.result-card');
    if (resultCard) {
        // Add a class to trigger animation
        setTimeout(() => {
            resultCard.classList.add('animated');
        }, 100);
    }
});