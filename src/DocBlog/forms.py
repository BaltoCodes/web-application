# forms.py
from django import forms

class OptionPricingForm(forms.Form):
    current_stock_price = forms.FloatField(label='Current Stock Price')
    strike_price = forms.FloatField(label='Strike Price')
    interest_rate = forms.FloatField(label='Interest Rate')
    volatility = forms.FloatField(label='Volatility')
    time_to_expiration = forms.FloatField(label='Time to Expiration')
    option_type = forms.ChoiceField(choices=[('call', 'Call Option'), ('put', 'Put Option')], label='Option Type')
