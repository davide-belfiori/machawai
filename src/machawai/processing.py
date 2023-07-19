"""
    Functions for curve processing.
"""

import numpy as np
from sklearn.metrics import r2_score

# -------------------
# --- BREAK POINT --- 
# -------------------

def derivata_seconda(X: 'list | np.ndarray', Y: 'list | np.ndarray') -> np.ndarray:
    """
        Calcola la derivata seconda di una funzione date 
        le sequenze di coordinate `X` ed `Y`.

            dY[i] = (Y[i+2] - 2*Y[i+1] + Y[i]) / (X[i+1] - X[i])^2 

        ATTENZIONE: `len(dY) = len(Y) - 2`.
    """
    f_len = len(Y)
    # la funzione deve essere definita in aleno 3 punti
    assert f_len >= 3
    der_len = f_len - 2
    # X ed Y devono avere la stessa lunghezza
    assert len(X) == f_len
    # inizializza la derivata come una sequenza di 0
    dY = [0.] * der_len
    for i in range(der_len):
        num = (Y[i+2] - 2*Y[i+1] + Y[i])
        den = (X[i+1] - X[i])**2
        if den == 0:
            dY[i] = np.Inf if num >= 0 else - np.Inf
        else:
            dY[i] = (Y[i+2] - 2*Y[i+1] + Y[i]) / (X[i+1] - X[i])**2
    return dY

def trova_sezione(Y: 'list | np.ndarray', p: float):
    """
        Calcola gli indici della sezione di interesse che 
        va da `Ymax` = massimo di Y fino ad `Ymax - p * Ymax`
    """
    # Calcola il punto iniziale della sezione come il massimo di Y
    start_point = np.argmax(Y)
    # Calcola il valore massimo di Y
    Ymax = np.max(Y)
    # Calcola il punto finale della sezione come Ymax - p% di Ymax
    Yp = Ymax - (p * Ymax)
    end_point = start_point
    for i in range(start_point + 1, len(Y)):
        if Y[i] < Yp:
            return start_point, end_point
        else:
            end_point = i
    return start_point, end_point

def max_var_point(der: 'list | np.ndarray'):
    """
        Restituisce il punto di massima variazione di una serie.
    """
    if len(der) <= 2:
        return 0
    max_var = der[1] - der[0]
    max_var_idx = 0
    for i in range(1, len(der) - 1):
        var = abs(der[i+1] - der[i])
        if var > max_var:
            max_var = var
            max_var_idx = i
    return max_var_idx

def trova_rottura(X: 'list | np.ndarray', Y: 'list | np.ndarray', p: float = 0.1):
    """
        Calcola l'indice del punto di rottura.
    """
    # Calcola derivata seconda dell'intera curva
    der2 = derivata_seconda(Y = Y, X = X)
    # Calcola la sezione di interesse
    start_section, end_section = trova_sezione(Y=Y, p=p)
    section_len = (end_section + 1) - start_section
    # Se la sezione di interesse è formata da 1 solo punto, quello è il punto di rottura
    if section_len <= 1:
        return start_section, (start_section, end_section)
    # Seleziona il tratto di derivata nella sezione di interesse
    der2_section = der2[start_section: end_section + 1]
    assert len(der2_section) + 2 >= end_section + 1 - start_section
    # Calcola il punto di massima variazione della derivata nella sezione di interesse
    break_point = max_var_point(der2_section)

    return start_section + break_point, (start_section, end_section)

# -------------
# --- SLOPE ---
# -------------

def calcola_r2(x, y, l_scan = 15):
    # Controlla che le liste abbiano la stessa lunghezza
    assert len(x) == len(y)
    
    # Controlla che le liste siano abbastanza lunghe
    assert len(x) >= l_scan

    r2_list = []
    coef_list = []
    y_pred_list = []
    x_pred_list = []
    for i in range(len(x) - l_scan - 1):
        # Prendi un segmento di 15 punti
        x_segment = x[i:i+l_scan]
        y_segment = y[i:i+l_scan]
        
        # Calcola la linea di regressione
        coef = np.polyfit(x_segment, y_segment, 1)
        y_pred = np.polyval(coef, x_segment)
        
        # Calcola l'R2 per il segmento
        r2 = r2_score(y_segment, y_pred)
        r2_list.append(r2)
        coef_list.append(coef)
        y_pred_list.append(y_pred)
        x_pred_list.append(x_segment)

    return r2_list, coef_list, x_pred_list, y_pred_list

def tratti_r2_sopra_soglia(r2_list, soglia=0.98, l_min = 20):
    tratti_sopra_soglia = []
    indici_sopra_soglia = []
    tratto_corrente = []
    indice_corrente = []
    
    for i, r2 in enumerate(r2_list):
        if r2 >= soglia:
            tratto_corrente.append(r2)
            indice_corrente.append(i)
        else:
            if tratto_corrente and len(tratto_corrente) >= l_min:  # se la lista non è vuota
                tratti_sopra_soglia.append(tratto_corrente)
                indici_sopra_soglia.append(indice_corrente)
            tratto_corrente = []  # ripartiamo da una nuova lista
            indice_corrente = []  # ripartiamo da una nuova lista
    
    # Aggiungi l'ultimo tratto e l'ultimo indice se non sono vuoti
    if tratto_corrente and len(tratto_corrente) >= l_min:
        tratti_sopra_soglia.append(tratto_corrente)
        indici_sopra_soglia.append(indice_corrente)
    
    return tratti_sopra_soglia, indici_sopra_soglia

def calcola_sottosegmenti_r2(indici, r2_tratto, x, y, l_min = 20):
    assert len(indici) == len(r2_tratto)
    assert len(x) == len(y)
    assert len(indici) >= l_min

    risultati = []
    
    for lunghezza in range(l_min, len(indici) + 1):  # partiamo da segmenti di lunghezza 15 fino alla lunghezza massima del tratto
        for i in range(len(indici) - lunghezza + 1):  # scorriamo il tratto per ottenere tutti i possibili sottosegmenti di questa lunghezza
            indici_sottosegmento = indici[i:i+lunghezza]
            x_sottosegmento = [x[j] for j in indici_sottosegmento]
            y_sottosegmento = [y[j] for j in indici_sottosegmento]
            
            # Calcoliamo la regressione lineare per il sottosegmento
            coef = np.polyfit(x_sottosegmento, y_sottosegmento, 1)
            y_pred = np.polyval(coef, x_sottosegmento)
            
            # Calcoliamo l'R2 per il sottosegmento rispetto alla curva originale
            r2 = r2_score(y_sottosegmento, y_pred)
            
            # Aggiungiamo il risultato alla lista
            risultati.append({
                'indici': indici_sottosegmento,
                'x_sottosegmento': x_sottosegmento,
                'y_pred': y_pred,
                'lunghezza': lunghezza,
                'r2': round(r2, 4)
            })

    return risultati

def filtra_e_trova_piu_lungo(risultati):
    # Filtra i risultati per tenere solo quelli con R2 almeno 0.995
    # filtrati = [r for r in risultati if r['r2'] >= 0.9998]

    # Calcola il massimo valore di r2 tra tutti i risultati
    max_r2 = max(r['r2'] for r in risultati)

    # Filtra i risultati per includere solo quelli con r2 uguale al valore massimo
    filtrati = [r for r in risultati if r['r2'] == max_r2]

    # Se non rimane nessun risultato, restituiamo None
    if not filtrati:
        return None

    # Trova e restituisci il risultato con la lunghezza massima
    return max(filtrati, key=lambda r: r['lunghezza'])

def trova_lineare(x: 'list | np.ndarray', y: 'list | np.ndarray', soglia: float = 0.98, l_scan: int = 15, l_min: int = 20):
    r2_list, coef_list, x_pred_list, y_pred_list = calcola_r2(x, y, l_scan=l_scan)
    tratti_sopra_soglia, indici_sopra_soglia = tratti_r2_sopra_soglia(r2_list, soglia=soglia, l_min=l_min)
    primo_tratto, indici_primo_tratto = tratti_sopra_soglia[0], indici_sopra_soglia[0]
    risultati = calcola_sottosegmenti_r2(indici_primo_tratto, primo_tratto, x, y, l_min=l_min)
    tratto_ottimo = filtra_e_trova_piu_lungo(risultati)
    start = tratto_ottimo['indici'][0]
    end = tratto_ottimo['indici'][-1]

    return start, end
