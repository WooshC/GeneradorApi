import requests

url = "http://127.0.0.1:5000/generar_musica"
data = {
    "note_count": 80,
    "speed_factor": 3.0
}

response = requests.post(url, json=data)
if response.status_code == 200:
    print("Música generada exitosamente. Puedes descargarla usando /descargar.")
else:
    print("Hubo un error al generar la música.")


url = "http://127.0.0.1:5000/descargar_midi"
response = requests.get(url, stream=True)

if response.status_code == 200:
    with open("music_generated.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Descarga completada.")
else:
    print("Error al descargar la música generada.")
