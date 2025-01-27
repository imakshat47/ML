import requests 
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://www.ambitionbox.com/list-of-companies?campaign=desktop_nav&page=1'
headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'} 
print(requests.get(url,headers=headers))