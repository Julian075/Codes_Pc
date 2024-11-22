import pandas as pd
from datetime import datetime,timedelta
import math

# Cargar el archivo CSV (reemplaza con tu archivo real)
input_file = '/home/julian/Descargas/Datos_Fototrampeo_2024.csv'
output_file = '/home/julian/Descargas/processed_Datos_Fototrampeo_2024.csv'

# Leer el archivo CSV
df = pd.read_csv(input_file, encoding="latin1", low_memory=False)

# Asegurarse de que las columnas no tengan espacios adicionales
df.columns = df.columns.str.strip()

# Mapeo de abreviaciones de meses a números
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


# Función para convertir un mes en número, si no es válido devuelve '--'
def convert_month(month):
    if isinstance(month, str):
        month = month.strip()
        return month_map.get(month[:3], '--')  # Extraer los primeros 3 caracteres y convertir
    return '--'


# Función para convertir una cadena a entero, si no es válido devuelve '--'
def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return '--'


# Función para convertir 'eventTime' a formato '2H 55M 08S'
def format_duration(event_time):
    try:
        h, m, s = map(int, event_time.split(':'))
        return f"{h}H {m}M {s}S"
    except (ValueError, AttributeError):
        return '--'

# Función para convertir 'eventTime' a total de segundos
def to_seconds(event_time):
    try:
        h, m, s = map(int, event_time.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None

# Función para convertir 'eventTime' a radianes
def to_radians(seconds):
    try:
        return (seconds / 86400) * 2 * math.pi  # Total seconds in a day = 86400
    except (TypeError, ZeroDivisionError):
        return None


# Procesar cada fila y aplicar las transformaciones
processed_rows = []

for index, row in df.iterrows():
    # Combinar Day, Month y Year para formar la fecha en el formato 'DD/MM/YYYY'
    day = safe_int(row['Day'])
    month=row['Month']
    if row['Month'] in month_map.keys():
        month = convert_month(row['Month'])
    month = safe_int(month)
    year = safe_int(row['Year'])
    photo_date = f"{day:02}/{month:02}/{year}" if day != '--' and month != '--' and year != '--' else '--'


    time_sec= to_seconds(row['eventTime'])
    time_std = to_seconds(row['eventTime'])
    time_rad = to_radians(time_std)

    # Convertir Time (duración) en formato "2H 45M 11S"
    time_duration = format_duration(row['eventTime'])

    # Crear una nueva fila procesada con los valores formateados
    processed_row = {
        'camera_id': row['eventID'],  # Renombrar columna
        'Species': row['species'],  # Renombrar columna
        'Photo.Date': photo_date,
        'Photo.Time': row['eventTime'],
        'Individuals': row['individualCount'],  # Renombrar columna
        'survey_id': row['locality'],  # Renombrar columna
        'class': row['class'],  # Renombrar columna
        'Time': time_duration,
        'time.sec': time_sec,
        'time.std': time_std,
        'time.rad': time_rad
    }

    # Añadir la fila procesada a la lista
    processed_rows.append(processed_row)

# Crear un DataFrame con las filas procesadas
processed_df = pd.DataFrame(processed_rows)

# Guardar el nuevo archivo CSV
processed_df.to_csv(output_file, index=False, encoding='latin1')

print(f"Archivo procesado guardado en {output_file}")
