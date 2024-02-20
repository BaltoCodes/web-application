document.addEventListener('DOMContentLoaded', function() { 
    //Renvoi vers la page d'accueil si on clique sur le wolf
    document.getElementById("icone").addEventListener("click", function() {
        window.location.href = "/thomas/";
    });
    document.getElementById("user").addEventListener("click", function() {
        window.location.href = "/dashboard/";
    });
    
    const radioButtons = document.querySelectorAll('input[type="checkbox"]');
    const secondForm = document.getElementById('secondForm');
    secondForm.addEventListener('submit', function(event) {
        const existingChart = Chart.getChart('secondChart');

        if (existingChart) {
            existingChart.destroy();
        }
        event.preventDefault(); 
        var filteredData=filterData();
        updateChart(filteredData);
    });
    

    function filterData() {
        /*var proportionOr = revenu*0.05*(risque/10);*/
        var revenu = parseFloat(document.getElementById('montant').value);
        var risque = parseFloat(document.getElementById('slider').value);
        var frequence = parseFloat(document.getElementById('sliderTwo').value);
        
        var or = document.getElementById('or').checked ? 1 : 0;
        var etf = document.getElementById('etf').checked ? 1 : 0;
        var actions = document.getElementById('actions').checked ? 1 : 0;
        var immobilier = document.getElementById('immobilier').checked ? 1 : 0;
        var cryptomonnaies = document.getElementById('cryptomonnaies').checked ? 1 : 0;



        var proportionOr = (revenu*(0.05)*or/(10.0001-risque))*10;
        var proportionActions = revenu*(0.25)*actions*(risque);
        var proportionCrypto = revenu*(0.25)*cryptomonnaies*(risque);  
        var proportionImmo = revenu*(0.1)*immobilier/(10-risque);
        var proportionEtf = revenu*(0.35)*etf*(10-risque);

        var total=proportionEtf+proportionImmo+proportionCrypto+proportionActions+proportionOr;


        var proportVariableOr = Math.round((proportionOr/total)* revenu);
        var proportVariableActions = Math.round((proportionActions/total)*revenu);
        var proportVariableImmo= Math.round((proportionImmo/total)*revenu);
        var proportVariableCrypto = Math.round((proportionCrypto/total)*revenu);
        var proportVariableEtf = Math.round((proportionEtf/total)*revenu);


        const filteredData = {
            labels: ['ETF / Indices', 'Cryptomonnaies', 'Actions', 'Immobilier', 'Matières primaires'],
            datasets: [{
                data: [etf === 1 ? proportVariableEtf : 0, 
                    cryptomonnaies === 1 ? proportVariableCrypto : 0, 
                    actions === 1 ? proportVariableActions : 0, 
                    immobilier === 1 ? proportVariableImmo : 0, 
                    or === 1 ? proportVariableOr :0 ],

                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)' , 
                    'rgba(54, 162, 235, 0.5)', 
                    'rgba(255, 206, 86, 0.5)' , 
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
            var filteredData=filterData();
            updateChart(filteredData);
        });
    });

    const sliderFrequence = document.getElementById('sliderTwo');
    sliderFrequence.addEventListener('input', function() {
        frequence = parseFloat(this.value);
        var filteredData=filterData();
        //filteredData.datasets[0].data[1] = frequence;
        updateChart(filteredData);
    });


    const sliderRisque = document.getElementById('slider');
    sliderRisque.addEventListener('input', function() {
        risque = parseFloat(this.value);   
        var filteredData=filterData();                 
        //filteredData.datasets[0].data[2] = risque;
        updateChart(filteredData);
    });


    function updateChart(filteredData) {
        const existingChart = Chart.getChart('secondChart');
        if (existingChart) {
            existingChart.destroy();
        }
        const ctx = document.getElementById('secondChart').getContext('2d');
        new Chart(ctx, {
            type: 'pie',    
            data: filteredData,
        });
    }
    
});