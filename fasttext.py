import requests
import os
import csv
import func

t, m = func.read(r'C:\Users\Gleb1\Desktop\Comments\Universities', 'all')
func.write_txt(t, m, 'all_t', r'C:\Users\Gleb1\Desktop\Comments\Universities')