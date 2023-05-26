from django.http import HttpResponse
import datetime

from django.shortcuts import render
import ctypes


def test(request):

    return render(request, "index.html")

def resultat(request):
    if request.method == 'POST':
        photo_data = request.POST['photoData']
    learning_emotions = ctypes.CDLL('C:/Users/dbelarbia/ESGI/pa/Projet-Annuel/target/release/learning_emotions.dll')
    my_add = learning_emotions.my_add
    my_add.argtypes = [ctypes.c_int, ctypes.c_int]
    my_add.restype = ctypes.c_int
    reponse = my_add(3, 2)
    print(reponse)
    return render(request, "index.html", context={"reponse": reponse})


