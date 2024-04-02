document.addEventListener('DOMContentLoaded', function() {
    const inputSentence = document.getElementById('inputSentence');
    const modelSelect = document.getElementById('modelSelect');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');

    predictButton.addEventListener('click', function() {
        const sentence = inputSentence.value;
        
        // Creating a request object to send to the backend API
        // Just replace '/api/predict' with the actual API endpoint URL
        console.log(sentence)
        var apiUrl = `/predict/${sentence}`

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
                console.log(data)

                var final_response = ""

                predictionResult.innerHTML = `
                <h3>Prediction :</h3>
                <h4><b>${data['prediction']}<br></b><h4/>
                </br>
                <h4> POS TAGS : </h4>
                ${data['pos_tags']}<br>
                </br>
                <h4> CODE_TAGS :</h4>
                ${data['code_tags']}<br>
                `;
                predictionResult.classList.add('show');
            })
            .catch(error => {
                predictionResult.textContent = 'Error: Unable to fetch prediction.';
                console.error(error);
            });
    });
});