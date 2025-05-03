"""
Common Utils Module
=================

Modul ini berisi fungsi-fungsi umum yang digunakan
di seluruh aplikasi.
"""

import os
import pandas as pd
import numpy as np
import datetime
import json

def save_to_csv(data, filename, index=True):
    """
    Menyimpan data ke file CSV
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        Data yang akan disimpan
    filename : str
        Nama file untuk menyimpan data
    index : bool, optional
        Apakah menyertakan indeks dalam output (default: True)
    """
    # Konversi ke DataFrame jika dict
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Pastikan direktori ada
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Simpan ke CSV
    data.to_csv(filename, index=index)
    print(f"Data saved to {filename}")

def save_to_json(data, filename):
    """
    Menyimpan data ke file JSON
    
    Parameters:
    -----------
    data : dict or list
        Data yang akan disimpan
    filename : str
        Nama file untuk menyimpan data
    """
    # Pastikan direktori ada
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Simpan ke JSON
    with open(filename, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)
    print(f"Data saved to {filename}")

def format_timestamp(dt=None):
    """
    Membuat string timestamp yang rapi
    
    Parameters:
    -----------
    dt : datetime, optional
        Objek datetime untuk diformat (default: datetime.now())
        
    Returns:
    --------
    str
        String timestamp (format: YYYY-MM-DD_HH-MM-SS)
    """
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

def format_number(num, decimal_places=2):
    """
    Memformat angka dengan pemisah ribuan dan n tempat desimal
    
    Parameters:
    -----------
    num : float
        Angka yang akan diformat
    decimal_places : int, optional
        Jumlah tempat desimal (default: 2)
        
    Returns:
    --------
    str
        String angka yang diformat
    """
    return f"{num:,.{decimal_places}f}"

def format_percentage(num, decimal_places=2):
    """
    Memformat angka sebagai persentase
    
    Parameters:
    -----------
    num : float
        Angka yang akan diformat (dalam desimal, mis. 0.1234 = 12.34%)
    decimal_places : int, optional
        Jumlah tempat desimal (default: 2)
        
    Returns:
    --------
    str
        String persentase yang diformat
    """
    return f"{num * 100:.{decimal_places}f}%"

def ensure_directory_exists(directory):
    """
    Memastikan direktori ada, membuat jika belum ada
    
    Parameters:
    -----------
    directory : str
        Path direktori yang akan diperiksa/dibuat
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

class NumpyEncoder(json.JSONEncoder):
    """
    JSON Encoder khusus untuk menangani tipe data NumPy
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj) 