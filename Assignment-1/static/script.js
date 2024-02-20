document.addEventListener('DOMContentLoaded', function() {
    const inputSentence = document.getElementById('inputSentence');
    const modelSelect = document.getElementById('modelSelect');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');

    predictButton.addEventListener('click', function() {
        const sentence = inputSentence.value;
        const selectedModel = modelSelect.value;
        
        // Creating a request object to send to the backend API
        // Just replace '/api/predict' with the actual API endpoint URL
        console.log(selectedModel)
        var apiUrl = `/${selectedModel}/${sentence}`

        console.log("apiUrl" , apiUrl)


        const request = new Request(apiUrl, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        fetch(request)
            .then(response => response.json())
            .then(data => {
                // handle the reesponse data here
                predictionResult.innerHTML = `<h3>Prediction :<h3><br> <b>${data['prediction']}</b>`;
                predictionResult.classList.add('show');
            })
            .catch(error => {
                predictionResult.textContent = 'Error: Unable to fetch prediction.';
                console.error(error);
            });
    });
});