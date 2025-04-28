// Tweet character counter
document.addEventListener('DOMContentLoaded', function () {
    const tweetText = document.getElementById('tweet_text');
    if (tweetText) {
        tweetText.addEventListener('input', function () {
            const charCount = this.value.length;
            const maxLength = 280; // Twitter's character limit

            // Update character count display
            let counter = document.getElementById('char-counter');
            if (!counter) {
                counter = document.createElement('small');
                counter.id = 'char-counter';
                counter.className = 'text-muted';
                this.parentNode.appendChild(counter);
            }

            counter.textContent = `${charCount}/${maxLength} characters`;

            // Add warning class if approaching limit
            if (charCount > maxLength * 0.9) {
                counter.classList.add('text-warning');
            } else {
                counter.classList.remove('text-warning');
            }
        });
    }

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function (event) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('is-invalid');
                } else {
                    field.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                event.preventDefault();
                alert('Please fill in all required fields');
            }
        });
    });
});

// Initialize AOS
AOS.init({
    duration: 800,
    easing: 'ease-in-out',
    once: true
});

// Character counter for tweet textarea
const tweetInput = document.getElementById('tweet');
const charCount = document.getElementById('charCount');
if (tweetInput && charCount) {
    tweetInput.addEventListener('input', function() {
        const currentLength = this.value.length;
        charCount.textContent = currentLength;
        if (currentLength > 280) {
            this.value = this.value.substring(0, 280);
            charCount.textContent = 280;
        }
    });
}

// Form submission handling
const form = document.getElementById('predictionForm');
if (form) {
    form.addEventListener('submit', function() {
        const submitButton = form.querySelector('button[type="submit"]');
        const spinner = submitButton.querySelector('.spinner-border');
        if (spinner) spinner.classList.remove('d-none');
        submitButton.disabled = true;
    });
}

// Initialize progress bars
window.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.progress-bar[data-width]').forEach(bar => {
        const width = bar.getAttribute('data-width');
        bar.style.width = `${width}%`;
    });
});



