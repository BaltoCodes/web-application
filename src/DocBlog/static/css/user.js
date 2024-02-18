document.addEventListener('DOMContentLoaded', function() { 
    //Renvoi vers la page d'accueil si on clique sur le wolf
    document.getElementById("icone").addEventListener("click", function() {
        window.location.href = "/thomas/";
    });
    document.getElementById("user").addEventListener("click", function() {
        window.location.href = "/dashboard/";
    });

});