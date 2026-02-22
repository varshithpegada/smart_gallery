from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Photo

class RegisterForm(UserCreationForm):
    # This inherits all standard fields (Username, Password, Confirm Password)
    class Meta(UserCreationForm.Meta):
        fields = UserCreationForm.Meta.fields + () # Add 'email' here if you want it

class PhotoUploadForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ['image']