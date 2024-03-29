# forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
#from .models import CustomUser


class OptionPricingForm(forms.Form):
    current_stock_price = forms.FloatField(label='Current Stock Price')
    strike_price = forms.FloatField(label='Strike Price')
    interest_rate = forms.FloatField(label='Interest Rate')
    volatility = forms.FloatField(label='Volatility')
    time_to_expiration = forms.FloatField(label='Time to Expiration')
    option_type = forms.ChoiceField(choices=[('call', 'Call Option'), ('put', 'Put Option')], label='Option Type')




class ProfilInvestisseur(forms.Form):
    revenu=forms.TextInput(label='revenu')
    epargne=forms.Textarea(label='epargne')
    total_asset=forms.NumberInput(label='total-asset')
    bur_investissement=forms.TextInput(label='but-investissement')
    risque=forms.TextInput(label='risque')


class ClasseActifs(forms.Form):
    montant=forms.NumberInput(label='montant')

    


