
from music21 import *
import io
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS  # Importa la extensión Flask-CORS
import os
import tensorflow as tf
import shutil
import tempfile
# Importando las bibliotecas necesarias
import numpy as np
import pandas as pd
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Importando los módulos necesarios de advertencias y estableciendo una semilla aleatoria
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Estableciendo la ruta de lilypond para la representación gráfica de la música
os.environ['lilypondPath'] = '/usr/bin/'
import tensorflow as tf

modelo_entrenado = tf.keras.models.load_model(r"C:\Users\USUARIO\Downloads\Melodias\modelMozart1.2.h5")
#modelo_entrenado_Grieg = tf.keras.models.load_model(r"C:\Users\USUARIO\Downloads\Melodias\modelogrieg.h5")
#modelo_entrenado_modelo = tf.keras.models.load_model(r"C:\Users\USUARIO\Downloads\Melodias\modelo1.1.h5")
length = 40
data = pd.read_csv(r'C:\Users\USUARIO\Downloads\Data\dataMozart1.csv')

# Convertir las columnas 'features' y 'targets' en matrices numpy
Xpri = np.array(data['features'].apply(eval).tolist())  # Usamos apply(eval) para convertir la lista en numpy array
ypri = np.array(data['targets'].apply(eval).tolist())   # Usamos apply(eval) para convertir la lista en numpy array

# Asegurarse de que X tenga la forma correcta (L_datapoints, length, 1)
Xpri = Xpri.reshape((Xpri.shape[0], length, 1))

#Taking out a subset of data to be used as seed
X_trainpri, X_valpri, y_trainpri, y_valpri = train_test_split(Xpri, ypri, test_size=0.2, random_state=25)

# Leer el corpus desde el archivo de texto plano
with open(r"C:\Users\USUARIO\Downloads\Data\corpus_cleaned_Mozart.txt", "r") as f:
    Corpus = [line.strip() for line in f]



def chords_n_notes(Snippet):
    Melody = []
    offset = 0 #Incremental
    for i in Snippet:
        #If it is chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".") #Seperating the notes in chord
            notes = []
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        # pattern is a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)
    return Melody_midi

Melody_Snippet = chords_n_notes(Corpus[:100])

# Almacenando todos los caracteres únicos presentes en mi corpus para construir un diccionario de mapeo.
symb = sorted(list(set(Corpus)))

L_corpus = len(Corpus) #Longitud del corpus
L_symb = len(symb) #Longitud total de caracteres únicos.

#Construyendo un diccionario para acceder al vocabulario a través de índices y viceversa
mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

length = 40
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])

L_datapoints = len(targets)



def Malody_Generator(Note_Count, speed_factor):
    
    seed = X_valpri[np.random.randint(0,len(X_valpri)-1)]
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,length,1)
        prediction = modelo_entrenado.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(L_symb)
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)
    # Reducir la duración de las notas para aumentar la velocidad
    for element in Melody_midi.recurse():
        if isinstance(element, note.Note) or isinstance(element, chord.Chord):
            element.duration.quarterLength *= speed_factor
    return Music, Melody_midi


app = Flask(__name__)
CORS(app, origins="http://localhost:5500")  # Solo permite solicitudes desde http://127.0.0.1:5500

@app.route('/a', methods=['GET'])
def index():
    return render_template('index.html')

music_counter = 1
midi_filename = ""

@app.route('/generar_musica', methods=['POST'])
def generar_musica():
    global music_counter, midi_filename

    data = request.json
    note_count = float(data.get('note_count', 50))  # Valor predeterminado de 50 si no se proporciona
    speed_factor = float(data.get('speed_factor', 1.0))  # Valor predeterminado de 1.0 si no se proporciona
    
    # Llamar a la función Malody_Generator
    _, melody_midi = Malody_Generator(note_count, speed_factor)
    
    # Generar el nombre del archivo MIDI con el contador
    midi_filename = f"generated_music_{music_counter}.mid"
    music_counter += 1  # Incrementar el contador para la próxima vez
    
    # Guardar el archivo MIDI
    midi_filepath = os.path.join(r"C:\Users\USUARIO\OneDrive - Escuela Politécnica Nacional\Escritorio\apis", midi_filename)
    melody_midi.write('midi', fp=midi_filepath)
    response = jsonify({"message": "Música generada exitosamente", "midi_file": midi_filename, "note_count": note_count, "speed_factor": speed_factor})
    #response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5500')  # Reemplaza con la dirección URL del archivo HTML
    return response

@app.get("/descargar_midi")
def descargar_midi():
    global midi_filename

    # Verificar que se haya generado previamente un archivo MIDI
    if not midi_filename:
        return "No se ha generado música previamente"

    midi_filepath = os.path.join(r"C:\Users\USUARIO\OneDrive - Escuela Politécnica Nacional\Escritorio\apis", midi_filename)

    return send_file(midi_filepath, as_attachment=True, mimetype="audio/midi")

@app.get("/descargar")
def descargar_folder():
    folder_path = r"C:\Users\USUARIO\Downloads\archive"
    folder_name = "archivos"  # Nombre de la carpeta y del archivo ZIP
    
    # Comprimir el folder en un archivo ZIP
    shutil.make_archive(folder_name, 'zip', folder_path)

    # Envía la respuesta con el archivo comprimido
    return send_file(folder_name + ".zip", as_attachment=True, mimetype="application/zip")

@app.route('/play_music')
def play_music():
    global midi_filename

    # Verificar que se haya generado previamente un archivo MIDI
    if not midi_filename:
        return "No se ha generado música previamente"

    midi_filepath = os.path.join(r"C:\Users\USUARIO\OneDrive - Escuela Politécnica Nacional\Escritorio\apis", midi_filename)

    return send_file(midi_filepath, mimetype="audio/midi")


if __name__ == "__main__":
    app.run(debug=True)
