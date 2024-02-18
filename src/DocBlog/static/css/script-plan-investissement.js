document.addEventListener('DOMContentLoaded', function() { 
    //Renvoi vers la page d'accueil si on clique sur le wolf
    document.getElementById("icone").addEventListener("click", function() {
        window.location.href = "/thomas/";
    });
    document.getElementById("user").addEventListener("click", function() {
        window.location.href = "/dashboard/";
    });
    



    //Lorsqu'on appuie sur le bouton, calcul de la première répartition d'actifs
    const secondForm = document.getElementById('secondForm');
    secondForm.addEventListener('submit', function(event) {
        const existingChart = Chart.getChart('secondChart');

        if (existingChart) {
            existingChart.destroy();
        }
        event.preventDefault(); 

        var revenu = parseFloat(document.getElementById('montant').value);
        var risque = parseFloat(document.getElementById('slider').value);
        var frequence = parseFloat(document.getElementById('sliderTwo').value);
        
        var or = document.getElementById('or').checked ? 1 : 0;
        var etf = document.getElementById('etf').checked ? 1 : 0;
        var actions = document.getElementById('actions').checked ? 1 : 0;
        var immobilier = document.getElementById('immobilier').checked ? 1 : 0;
        var cryptomonnaies = document.getElementById('cryptomonnaies').checked ? 1 : 0;

        var nbClicked = or + etf + actions + immobilier + cryptomonnaies;
 

        var proportionOr = revenu*(0.05)*or;
        var proportionActions = revenu*(0.25)*actions;
        var proportionCrypto = revenu*(0.25)*cryptomonnaies;  
        var proportionImmo = revenu*(0.1)*immobilier;
        var proportionEtf = revenu*(0.35)*etf;

        

        var proportVariableOr = proportionOr + revenu*((100- (proportionEtf+proportionImmo+proportionCrypto+proportionActions+proportionOr))/(nbClicked*100));
        var proportVariableActions = proportionActions + revenu*((100- (proportionEtf+proportionImmo+proportionCrypto+proportionActions+proportionOr))/(nbClicked*100));
        var proportVariableImmo= proportionImmo + revenu*((100- (proportionEtf+proportionImmo+proportionCrypto+proportionActions+proportionOr))/(nbClicked*100));
        var proportVariableCrypto = proportionCrypto + revenu*((100- (proportionEtf+proportionImmo+proportionCrypto+proportionActions+proportionOr))/(nbClicked*100));
        var proportVariableEtf = proportionEtf + revenu*((100- (proportionEtf+proportionImmo+proportionCrypto+proportionActions+proportionOr))/(nbClicked*100));

     

        const radioButtons = document.querySelectorAll('input[type="radio"]');


        function filterData(or, immobilier, actions, cryptomonnaies, etf, risque, proportVariableActions, proportVariableCrypto, proportVariableEtf, proportVariableImmo, proportVariableOr, nbClicked) {
            /*var proportionOr = revenu*0.05*(risque/10);*/
            const filteredData = {
                labels: ['ETF / Indices', 'Cryptomonnaies', 'Actions', 'Immobilier', 'Matières primaires'],
                datasets: [{
                    data: [etf === 1 ? proportVariableEtf : 0, 
                        cryptomonnaies === 1 ? proportVariableCrypto : 0, 
                        actions === 1 ? proportVariableActions : 0, 
                        immobilier === 1 ? proportVariableImmo : 0, 
                        or === 1 ? proportVariableOr :0 ],

                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)' , // Rouge
                        'rgba(54, 162, 235, 0.5)', // Bleu
                        'rgba(255, 206, 86, 0.5)' , // Jaune
                        'rgba(75, 192, 192, 0.5)', 
                        'rgba(255, 75, 75, 0.736)' ,
                        'rgba(86, 230, 233, 0.736)'
                    ],
                    borderWidth: 1
                }]
            };

            return filteredData;
        }

        radioButtons.forEach(function(radioButton) {
            radioButton.addEventListener('change', function() {
                const id = this.id;
                const value = this.checked ? 1 : 0;

                if (id === 'or') {
                    or = value;
                    nbClicked+=1;

                } else if (id === 'immobilier') {
                    immobilier = value;
                    nbClicked+=1;

                } else if (id === 'actions') {
                    actions = value;
                    nbClicked+=1;

                } else if (id === 'cryptomonnaies') {
                    cryptomonnaies = value;
                    nbClicked+=1;

                } else if (id === 'etf') {
                    etf = value;
                    nbClicked+=1;

                }
                updateChart();
            });
        });

        const filteredData = filterData(or, immobilier, actions, cryptomonnaies, etf, risque,  proportVariableActions, proportVariableCrypto, proportVariableEtf, proportVariableImmo, proportVariableOr, nbClicked);

        const config = {
            type: 'pie',
            data: filteredData,
        };
        
        //Pie Chart
        const ctx = document.getElementById('secondChart').getContext('2d');
        new Chart(ctx, config);



        //Update en bougeant les sliders
        const sliderFrequence = document.getElementById('sliderTwo');
        sliderFrequence.addEventListener('input', function() {
            frequence = parseFloat(this.value);
            filteredData.datasets[0].data[1] = frequence;
            updateChart();
        });


        const sliderRisque = document.getElementById('slider');
        sliderRisque.addEventListener('input', function() {
            risque = parseFloat(this.value);                    
            filteredData.datasets[0].data[2] = risque;
            updateChart();
        });



        function updateChart() {
            const existingChart = Chart.getChart('secondChart');
            if (existingChart) {
                existingChart.destroy();
            }
            var nbClicked = or + etf + actions + immobilier + cryptomonnaies;
            const filteredData = filterData(or, immobilier, actions, cryptomonnaies, etf, risque,  proportVariableActions, proportVariableCrypto, proportVariableEtf, proportVariableImmo, proportVariableOr, nbClicked);


            const ctx = document.getElementById('secondChart').getContext('2d');
            new Chart(ctx, {
                type: 'pie',    
                data: filteredData,
            });
        }

    });
});