from django.http import HttpResponse
import datetime

from django.shortcuts import render
import ctypes
<<<<<<< HEAD
import numpy as np
=======

>>>>>>> 199dda4bafb9ccfaad99f4f3118679149ba9f719

def test(request):

    return render(request, "index.html")

def resultat(request):
<<<<<<< HEAD
    global photo_data
    if request.method == 'POST':
        photo_data = request.POST['photoData']
        learning_emotions = ctypes.CDLL('C:/Users/dbelarbia/ESGI/pa/Projet-Annuel/target/release/learning_emotions.dll')
        my_add = learning_emotions.my_add
        my_add.argtypes = [ctypes.c_int, ctypes.c_int]
        my_add.restype = ctypes.c_int
        reponse = my_add(3, 2)
        print(reponse)

        new_resized_image = photo_data
        learning_emotions.flatten_images.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
        learning_emotions.flatten_images.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1,
                                                                       flags='C_CONTIGUOUS')
        new_flattened_image = learning_emotions.flatten_images(new_resized_image)
        # inputs = learning_emotions.new_flattened_image.clone()
        predicted_class = learning_emotions.network.predict(new_flattened_image)

        classes = ["triste", "heureuse", "enervée"]
        predicted_class_name = classes[predicted_class]
    return render(request, "index.html", context={"reponse": predicted_class_name})
=======
    if request.method == 'POST':
        photo_data = request.POST['photoData']
    learning_emotions = ctypes.CDLL('C:/Users/dbelarbia/ESGI/pa/Projet-Annuel/target/release/learning_emotions.dll')
    my_add = learning_emotions.my_add
    my_add.argtypes = [ctypes.c_int, ctypes.c_int]
    my_add.restype = ctypes.c_int
    reponse = my_add(3, 2)
    print(reponse)
    return render(request, "index.html", context={"reponse": reponse})
>>>>>>> 199dda4bafb9ccfaad99f4f3118679149ba9f719


