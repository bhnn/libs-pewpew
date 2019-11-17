import csv
import platform
import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

project_path = './ml'

def read_mineral_list(file_path):
    """
    Reads list of all minerals, checks them for possible mistakes, reformats into easily iterable list
    """
    print('BEGIN Read list of all minerals, verify data...')
    with open(file_path, newline='') as file:
        f = csv.reader(file)
        
        ids = list()
        names = list()
        final = list()
        for num, name, elements, composition, m_class, m_group in f:
            # verify indexing integrity
            if ids:
                assert int(num) - int(ids[-1]) == 1, f'indexing gap found for \'{name}\' ({num})'
            
            assert num not in ids, f'duplicate ids found for \'{name}\' ({num})'
            ids.append(num)

            # check for duplicate elements
            assert name not in names, f'duplicate element names found for \'{name}\' ({num})'
            names.append(name)

            # verify amount of elements matches amount of compositions
            elements = elements.strip('[]').replace(' ', '').split(',')
            composition = [float(i) for i in composition.strip('[]').replace(']','').split(',')]
            assert len(elements) == len(composition), f'mismatch in amount of elements/compositions found for \'{name}\' ({num})'

            # verify sum of composition is valid
            assert 99.9 < sum(composition) < 100.999, f'invalid sum of atomic composition for \'{name}\' ({num})'

            # add asserted element to final output
            final.append((num, name, elements, composition, m_class, m_group))
        
        print('DONE! Reading and verifying minerals.')
        return final

def fix_sum_to_100(composition, method=0):
    """
    Fixes sum of passed atomic composition to sum up to 100.0 
    """
    if method == 0:
        # distribute difference of sum to 100.0 evenly across all composing elements
        if sum(composition) != 100:
            diff = (sum(composition) - 100) / len(composition)
            for i in range(len(composition)-1):
                composition[i] -= diff
            composition[-1] -= sum(composition) - 100
    elif method == 1:
        # subtract difference of sum to 100.0 from largest composing element
        if sum(composition) != 100:
            index = composition.index(max(composition))
            composition[index] -= sum(composition) - 100
    return composition

def assemble_nist_url(elements, percentages, low_wl=180, high_wl=960, resolution=1000, temp='1', eden='1e17'):
    """
    Assembles NIST LIBS link from elements and atomic composition.
    """
    print('  BEGIN Assemble NIST LIBS URL...')
    assert type(elements) is list
    assert type(percentages) is list
    content_len = len(elements)
    # base URL
    request_url = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl'
    # encoded composition string of all composing elements: El1:Perc1;El2:Perc2;...
    comp_string = '?composition=' + '%3B'.join([f'{elements[i]}%3A{percentages[i]}' for i in range(content_len)])
    # encoded spectra string for all composing elements: El10-2,El20-2,El30-2,...
    spec_string = '&spectra=' + '%2C'.join([f'{elements[i]}0-2' for i in range(content_len)])

    for i in range(content_len):
        request_url += '{}&mytext%5B%5D={}&myperc%5B%5D={}{}'.format(comp_string if i == 0 else '', 
                                                                     elements[i], 
                                                                     percentages[i], 
                                                                     spec_string if i == 0 else '')

    request_url += f'&low_w={low_wl}&limits_type=0&upp_w={high_wl}&show_av=2&unit=1&resolution={resolution}'
    request_url += f'&temp={temp}&eden={eden}&libs=1'
    print('  DONE! Assembling NIST LIBS URL.')
    return request_url

def get_webdriver():
    """
    Construct Chrome webdriver for Selenium.
    """
    print('BEGIN Get Selenum webdriver...')
    chrome_options = Options()
    # headless browser, no gui
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-extentions')

    driver_path = 'lib/chromedrivers/cd'
    
    if platform.system() == 'Linux':
        driver_path += '_linux'
    elif platform.system() == 'Darwin':
        driver_path += '_osx'
    elif platform.system() == 'Windows':
        driver_path += '_win.exe'
    else:
        raise ValueError('Chromedriver encountered invalid platform information!')

    driver = webdriver.Chrome(driver_path, options=chrome_options)
    print('DONE! Getting Selenium webdriver.')
    return driver

def retrieve_nist_data(driver, link, resolution=1000):
    """
    Open NIST LIBS link, recalculate resolution to specified value, extract spectral information.
    """
    print('  BEGIN Retrieve NIST data using Selenium...')
    # open generated link in browser
    driver.get(link)

    # wait for spectrum to load and the resolution textbox to unlock
    res_input = WebDriverWait(driver, 240).until(
        EC.presence_of_element_located((By.ID, 'resolution'))
    )

    # remove all content from it, then enter desired resolution and submit
    res_input.send_keys(Keys.CONTROL + 'a')
    res_input.send_keys(Keys.DELETE)
    res_input.send_keys(resolution)
    res_input.send_keys(Keys.ENTER)

    # wait until calculation is complete, then click Download CSV button
    csv_btn = WebDriverWait(driver, 240).until(
        EC.presence_of_element_located((By.NAME, 'ViewDataCSV'))
    )
    csv_btn.click()

    # switch to new tab with CSV data (js generates it with fixed name), then grab all text and close
    driver.switch_to.window('libs_data')
    text = BeautifulSoup(driver.page_source, 'lxml').text
    start = text.find('Wavelength')
    text = text[start:]
    
    print('  DONE! Retriebing NIST data.')
    return text

def clean_nist_data(data, use_header=False):
    """
    Removes unnecessary spectral information and reformats from string to list.
    """
    print('  BEGIN Clean and format NIST data...')
    # data is read as string, has to be converted into list of lists
    # only keep first 2 columns (wavelength + sum intensity)
    split_data = [line.split(',')[:2] for line in data.split('\n')[int(not use_header):]]
    print('  DONE! Cleaning formatting NIST data.')
    return split_data

def write_to_csv(data, filename, delimiter=','):
    """
    Write collected information to file.
    """
    print('  BEGIN Write data to .csv...')
    with open(filename, 'w+') as f:
        for line in data:
            f.write('{}\n'.format(','.join(line)))
        # writer = csv.writer(f, delimiter)
        # writer.writerows(data)
    print('  DONE! Writing data to .csv.')

def reset_webdriver(driver):
    """
    Resets Selenium webdriver between elements.
    """
    # clear out cookies -- not sure if needed
    driver.delete_all_cookies()

    # get list of current tabs
    current_tab = driver.current_window_handle
    tab_list = driver.window_handles
    tab_list.remove(current_tab)
    
    # close all but one open tabs from last session
    for tab in tab_list:
        driver.switch_to.window(tab)
        driver.close()
    
    driver.switch_to.window(current_tab)
    driver.get('about:blank')
    return driver

def calculate_alterations(composition, quantity):
    """
    Calculates quantity many alterations of the passed atomic composition.
    """
    alterations = composition
    # 1)
    #   take 3% relative from 1 random element
    #   give amount to 1st random element (up to 3% relative to its current size)
    #   give amount to next random element, etc, until taken 3% are used up
    # 2)
    #   roll random 0.0 - 3.0 (relative) to take from all elements in a row, randomised
    #   when sum of 3% taken is reached, roll random 0.0 - 3.0 (relative) to give back 
    #   to all elements in a row, randomised
    #   check that small elements dont get more than 3% (relative)

    return alterations

def create_synthetic_dataset():
    """
    Main routine to generate synthetic dataset.
    """
    print('BEGIN Create synthetic mineral dataset...')
    # read in list of all elements and verify integrity
    all_minerals = read_mineral_list('data/synthetic_minerals_raw_final.csv')

    # webdriver for NIST data access, persistent outside of loop to cut down load times of instance creation
    driver = get_webdriver()
    for num, name, elements, composition, m_class, m_group in all_minerals:
        print(f'BEGIN Next element: {name}')
        # calculate random alterations of composition
        alterations = calculate_alterations(composition, 10)
        for a in ['1']:
            # make sum of composing elements be exactly 100
            fix_sum_to_100(composition, method=0)

            # create NIST link, collect data
            link = assemble_nist_url(elements, composition, resolution=1000, temp=1, eden='1e17')
            # clean up results, discard columns
            data = retrieve_nist_data(driver, link, 1000)
            data = clean_nist_data(data)
            # save to file
            write_to_csv(data, 'results/placeholder_filename.csv')

            # reset webdriver for next iteration
            driver = reset_webdriver(driver)
            print(f'DONE! with element {name}')

    # clean up
    driver.quit()

    print('DONE! Creating synthetic mineral dataset.')

start_time = time.time()
create_synthetic_dataset()
# end_time = 
print(f'Runtime: {time.time() - start_time:.2f} seconds')

# todo
# - add permutations

# test https://physics.nist.gov/cgi-bin/ASD/lines1.pl?composition=H%3A28.13%3BC%3A0.78%3BNa%3A1.17%3BMg%3A1.48%3BAl%3A3.67%3BSi%3A9.38%3BCa%3A3.52%3BFe%3A2.66%3BO%3A49.209999999999994&mytext%5B%5D=H&myperc%5B%5D=28.13&spectra=H0-2%2CC0-2%2CNa0-2%2CMg0-2%2CAl0-2%2CSi0-2%2CCa0-2%2CFe0-2%2CO0-2&mytext%5B%5D=C&myperc%5B%5D=0.78&mytext%5B%5D=Na&myperc%5B%5D=1.17&mytext%5B%5D=Mg&myperc%5B%5D=1.48&mytext%5B%5D=Al&myperc%5B%5D=3.67&mytext%5B%5D=Si&myperc%5B%5D=9.38&mytext%5B%5D=Ca&myperc%5B%5D=3.52&mytext%5B%5D=Fe&myperc%5B%5D=2.66&mytext%5B%5D=O&myperc%5B%5D=49.209999999999994&low_w=180&limits_type=0&upp_w=960&show_av=2&unit=1&resolution=1000&temp=1&eden=1e17&libs=1
