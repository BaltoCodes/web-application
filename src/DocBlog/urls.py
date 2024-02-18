"""DocBlog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from .views import index, graph_view, calculator_view, get_message, world_is_yours, interactive_graph,spotify, rendement_wealth_management,  spotify_callback, obtenir_login, hull, world, thomas, plan_investissement, formation, dashboard, opportunites
from . import views


urlpatterns = [
    path('', world, name="index"),
    path('accueil/', world, name="index"),
    path('world/', world, name="world" ),
    path('admin/', admin.site.urls),
    path('graph/', graph_view, name="graph"), #Utilisé, fonctionnel
    path('chatbot/', get_message, name="get_message"), #Utilisé
    path('accueil/', world_is_yours, name="world"), #Utilisé fonctionnel mais version mieux existe
    path('interactive_graph/', interactive_graph, name="Graphique dz" ), #Utilisé mais pas fini 
    path('spotify/', spotify, name="Get my spoti"), #Utilisé fonctionnel 
    path('callback/', spotify_callback, name="callback"), #Utilisé fonctionnel
    path('obtenir_login/', obtenir_login, name="Les chiffres les vrais"), #Utilisé fonctionnel
    path('my-wealth/', rendement_wealth_management, name="my-wealth"),
    path('hull/', hull, name="hull"), 
    path('thomas/', thomas, name="thomas"),
    path('plan-investissement/', plan_investissement, name="plan_investissement"),
    path('me-former/', formation, name="Me former"),
    path('dashboard/', dashboard, name="Dashborad"),
    path('dashboard/opportunites', opportunites, name="Opportunités")

    #path('create-user/', create_user, name="Create new user")
]
